from operator import index
from numpy.lib.arraypad import _view_roi
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math
import os

from torch.nn.modules import dropout

from torchsummary import summary
from transformers.utils.dummy_pt_objects import AlbertModel

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel 
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from transformers.models.convbert.modeling_convbert import ConvBertConfig, ConvBertEncoder,ConvBertModel
from transformers.models.roberta.modeling_roberta import RobertaConfig,RobertaEncoder,RobertaModel
from transformers.models.albert.modeling_albert import AlbertAttention, AlbertTransformer, AlbertModel
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers import BertPreTrainedModel

import gc

import re

# 공통 블럭
class CommonModule(nn.Module):
    def __init__(self,args):
        super(CommonModule,self).__init__()
        self.args=args # 입력한 argument
        self.device = self.args.device # device

        self.max_seq_len = self.args.max_seq_len

        self.hidden_dim = self.args.hidden_dim//2 # 모델에 기본적으로 적용할 hidden_dim
        self.n_layers = self.args.n_layers # layer 수
        self.n_heads = self.args.n_heads # 어텐션에 적용할 head 수
        self.drop_out = self.args.drop_out # drop_out

        self.n_cate_cols = len(self.args.cate_cols) # 범주형 컬럼 수(answerCode 제외)
        self.n_cont_cols = len(self.args.cont_cols) # 연속형 컬럼 수

        self.cate_cols = self.args.cate_cols # 범주형 컬럼
        self.cont_cols = self.args.cont_cols # 연속형 컬럼

        self.concat_reverse = self.args.concat_reverse
        self.embedding_interaction = nn.Embedding(3,self.hidden_dim//2)
        self.embedding_cate = nn.ModuleList([
            nn.Embedding(v+1,self.hidden_dim//2)
            for k,v in self.args.cate_dict.items()
        ])
        self.n_heads=args.n_heads
        
        # 범주형 hidden dimension 공식
        self.n_embedding_hidden_dim = ((self.hidden_dim//2)*(self.n_cate_cols))


        print(f'범주형 : {self.args.cate_cols}')
        print(f'연속형 : {self.args.cont_cols}')

        # 연속형 전용 layer - input 차원 (?,?,self.cols_cnt) -> output (?,?,self.cm.hidden_dim//2) 
        self.embedding_cont = nn.Sequential(
            nn.Linear(1,self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2)
        )

        self.fc = nn.Linear((self.hidden_dim//2)*(self.n_cate_cols), 1)
        self.activation = nn.Sigmoid()

    def chk_type(self, cols):
        for i in cols:
            print(i.dtype)

    # input을 받고, 범주형 임베딩, 연속형을 각각 다른 레이어를 거친 후 concat
    def embed(self,input):
        # input의 순서는 (범주형),(mask),(interaction),(연속형),(gather_index)
        # 범주형의 차원은 (batch_size, max_seq_len), 연속형의 차원은 (batch_size, max_seq_len, 연속형 컬럼 개수)
        
        cate_cols = [input[i] for i in range(self.n_cate_cols)] # 범주형
        mask = input[self.n_cate_cols+1] # 마스크
        interaction = input[self.n_cate_cols+2] # interaction

        cont_cols = [input[i] for i in range(self.n_cate_cols+3,len(input)-1)] # 연속형
        gather_index = input[-1]
        batch_size = len(input[0])

        # self.chk_type(cate_cols) # type check

        embed_interaction = self.embedding_interaction(interaction)

        embed_cate = [self.embedding_cate[i](cate_cols[i]) 
        for i in range(len(cate_cols))]

        # 연속형 - 각자 다른 layer를 거친 후 concat
        # (batch, max_seq_len, 1)
        embed_cont_tmp = [self.embedding_cont(v) for i,v in enumerate(cont_cols)]
        if len(embed_cont_tmp)>0:
            embed_cont = torch.cat(embed_cont_tmp,2)
            embed_cont = embed_cont.view(batch_size,self.args.max_seq_len,-1)
        else:
            embed_cont = None
        # del input
        gc.collect()
        # embed_cate의 차원 : (bs, mas_seq_len, hidden//2)
        # return
        # (임베딩된 범주형 feature), mask, (임베딩 된 정답여부), (임베딩 된 연속형), gather_index
        return embed_cate,mask,embed_interaction,embed_cont,gather_index

    def future_mask(self,max_seq_len):
        future_mask = np.triu(np.ones((max_seq_len, max_seq_len)), k=1).astype('bool')
        return torch.from_numpy(future_mask)

######## Last Query
class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device
        self.cm = CommonModule(args)
        self.hidden_dim = self.cm.hidden_dim//2

        # 범주형 임베딩을 projection이나, q,k,v 에 쓰일 예정
        self.hidden_dim_used =  (self.hidden_dim)*(len(self.cm.cate_cols)) # (입력 hidden_dim)//2 * (범주형 컬럼 수)

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = self.cm.embedding_interaction
        self.embedding_cate = self.cm.embedding_cate
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        
        # encoder combination projection
        self.cate_proj = nn.Linear(self.hidden_dim_used, self.hidden_dim_used)


        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # Position-Embedding 사용여부
        self.is_position_embedding = False
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim_used)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim_used, out_features=self.hidden_dim_used)
        self.key = nn.Linear(in_features=self.hidden_dim_used, out_features=self.hidden_dim_used)
        self.value = nn.Linear(in_features=self.hidden_dim_used, out_features=self.hidden_dim_used)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim_used, num_heads=self.args.n_heads, dropout=self.cm.drop_out)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = FFN(self.args,self.hidden_dim_used)      

        self.ln1 = nn.LayerNorm(self.hidden_dim_used)
        self.ln2 = nn.LayerNorm(self.hidden_dim_used)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim_used,
            self.hidden_dim_used,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim_used, 1)
       
        self.activation = nn.Sigmoid()

    def get_mask(self, seq_len, index, batch_size):
        """
        batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        
        참고로 (batch_size*self.args.n_heads, seq_len, seq_len) 가 아니라
              (batch_size*self.args.n_heads,       1, seq_len) 로 하는 이유는
        
        last query라 output의 seq부분의 사이즈가 1이기 때문이다
        """
        # [[1], -> [1, 2, 3]
        #  [2],
        #  [3]]
        index = index.view(-1)

        # last query의 index에 해당하는 upper triangular mask의 row를 사용한다
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))
        mask = mask[index]

        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size, hidden_dim):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):

        # test, question, tag, _, mask, interaction, index = input

        # batch, seq_len 설정
        batch_size = input[0].size(0)
        seq_len = input[0].size(1)
        
        #common module 참고
        embed_cate,mask,embed_interaction,embed_cont,gather_index = self.cm.embed(input)

        # 신나는 embedding -> 그런거 없다
        embed = torch.cat(embed_cate, 2)
        # print(embed.shape)
        embed = self.cate_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        if self.is_position_embedding: # 기본값은 False
            position = self.get_pos(seq_len).to(self.args.device)
            embed_pos = self.embedding_position(position)
            embed = embed + embed_pos

        ####################### ENCODER #####################

        if self.args.lq_padding=='post':

            q = self.query(embed)

            # 이 3D gathering은 머리가 아픕니다. 잠시 머리를 식히고 옵니다.
            q = torch.gather(q, 1, gather_index.repeat(1, self.hidden_dim).unsqueeze(1))
            q = q.permute(1, 0, 2)

            k = self.key(embed).permute(1, 0, 2)
            v = self.value(embed).permute(1, 0, 2)

            ## attention
            # last query only
            self.mask = self.get_mask(seq_len, index, batch_size).to(self.device)
            out, _ = self.attn(q, k, v, attn_mask=self.mask)
        elif self.args.lq_padding=='pre':
            q = self.query(embed).permute(1, 0, 2)
            q = self.query(embed)[:, -1:, :].permute(1, 0, 2)

            k = self.key(embed).permute(1, 0, 2)
            v = self.value(embed).permute(1, 0, 2)

            ## attention
            # last query only
            out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size, self.hidden_dim_used)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################

        out = out.contiguous().view(batch_size, -1, self.hidden_dim_used)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

'''
src = "https://www.kaggle.com/gannonreynolds/saint-riid-0-798"
'''
class FFN(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop_out_layer = nn.Dropout(args.drop_out)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.drop_out_layer(x)

class GeneralizedSaintPlus(nn.Module):
    
    def __init__(self, args):
        super(GeneralizedSaintPlus, self).__init__()
        self.args = args
        self.cm = CommonModule(args)
        self.device = self.cm.device
        self.max_seq_len = self.cm.args.max_seq_len
        self.drop_out = self.cm.drop_out
        self.n_layers = self.cm.n_layers

        self.pos_embedding = nn.Embedding(self.max_seq_len,self.cm.hidden_dim//2)
        # Positional encoding
        self.pos_encoder = PositionalEncoding((self.cm.hidden_dim//2)*(self.cm.n_cate_cols), self.drop_out, self.cm.max_seq_len)
        self.pos_decoder = PositionalEncoding((self.cm.hidden_dim//2)*(self.cm.n_cate_cols), self.drop_out, self.cm.max_seq_len)

        self.transformer = nn.Transformer(
            d_model=(self.cm.hidden_dim//2)*1,
            nhead=self.cm.n_heads,
            num_encoder_layers=self.n_layers, 
            num_decoder_layers=self.n_layers,
            dropout=self.drop_out, 
            activation='relu')

        self.drop_out_layer = nn.Dropout(self.drop_out)
        self.fc = nn.Linear(self.cm.hidden_dim//2,1) # Linear Layer
        self.activation = self.cm.activation
        self.layer_norm = nn.LayerNorm((self.cm.hidden_dim//2)*1)
        self.ffn = FFN(self.cm.args,(self.cm.hidden_dim//2)*1)

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

        # 차원 측정용
        self.flag = False
    
    def get_mask(self, max_seq_len):
        
        future_mask = np.triu(np.ones((max_seq_len, max_seq_len)), k=1).astype('bool')
        
        return torch.from_numpy(future_mask)

    def forward(self, input):

        # batch, seq_len 설정
        batch_size = input[0].size(0)
        seq_len = input[0].size(1)
        
        #common module 참고
        embed_cate,mask,embed_interaction,embed_cont,gather_index = self.cm.embed(input)

        pos_id = torch.arange(seq_len).unsqueeze(0) # 1, max_seq_len
        tmp = [pos_id for i in range(batch_size)]
        pos_id = torch.cat(tmp,0)
        del tmp
        del embed_cont
        pos_id=pos_id.to(self.device)
        pos_id = self.pos_embedding(pos_id)

        if not self.flag:
            self.flag = True
            # print(f'pos_id_shape : {pos_id.shape}')
            # print(f'embed_cat : {embed_cate[0].shape}')

        enc = pos_id
        dec = pos_id
        
        for cate in embed_cate[:len(embed_cate)-1]:
            enc += cate
        
        # DECODER

        # cat_list = list(embed_cate)
        cat_list = list(embed_cate[len(embed_cate)-1:])
        cat_list.append(embed_interaction)
        for cate in cat_list:
            dec +=cate
        del cat_list

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = enc.permute(1, 0, 2)
        # print(f'embed_enc shape: {embed_enc.shape}')
        embed_dec = dec.permute(1, 0, 2)
        # print(f'embed_dec shape: {embed_dec.shape}')
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)
        # out = self.layer_norm(out)
        out = out.permute(1, 0, 2)
        # out = out.contiguous().view(batch_size, -1, (self.cm.hidden_dim//2)*(self.cm.n_cate_cols))
        x = self.ffn(out)
        # x = self.layer_norm(x+out)
        x=x+out
        # print(f'{x.shape}')
        out = self.fc(x)

        preds = self.activation(out).view(batch_size, -1)

        return preds
        # print(out.squeeze(-1))
        # return out.squeeze(-1)


# Saint Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class GeneralizedSaint(nn.Module):
    
    def __init__(self, args):
        super(GeneralizedSaint, self).__init__()
        
        self.cm = CommonModule(args)
        self.device = self.cm.device
        self.drop_out = self.cm.drop_out
        self.n_layers = self.cm.n_layers
        # encoder combination projection
        self.enc_comb_proj = nn.Sequential(
                                    nn.Linear((self.cm.hidden_dim//2)*
                                    (self.cm.n_cate_cols), (self.cm.hidden_dim//2)*(self.cm.n_cate_cols)),
                                    # nn.LayerNorm(self.cm.hidden_dim)
                            )

        # DECODER embedding
        # decoder combination projection
        self.dec_comb_proj = nn.Sequential(
                                    nn.Linear((self.cm.hidden_dim//2)*
                                    (self.cm.n_cate_cols+1), (self.cm.hidden_dim//2)*(self.cm.n_cate_cols)),
                                    # nn.LayerNorm(self.cm.hidden_dim)
                            )

        # Positional encoding
        self.pos_encoder = PositionalEncoding((self.cm.hidden_dim//2)*(self.cm.n_cate_cols), self.drop_out, self.cm.max_seq_len)
        self.pos_decoder = PositionalEncoding((self.cm.hidden_dim//2)*(self.cm.n_cate_cols), self.drop_out, self.cm.max_seq_len)

        self.transformer = nn.Transformer(
            d_model=(self.cm.hidden_dim//2)*(self.cm.n_cate_cols),
            nhead=self.cm.n_heads,
            num_encoder_layers=self.n_layers, 
            num_decoder_layers=self.n_layers, 
            dim_feedforward=self.cm.hidden_dim, 
            dropout=self.drop_out, 
            activation='relu')

        self.fc = self.cm.fc # Linear Layer
        self.activation = self.cm.activation

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):

        # batch, seq_len 설정
        batch_size = input[0].size(0)
        seq_len = input[0].size(1)
        
        #common module 참고
        embed_cate,mask,embed_interaction,embed_cont,gather_index = self.cm.embed(input)
        
        embed_cate_enc = torch.cat(embed_cate, 2)

        embed_enc = self.enc_comb_proj(embed_cate_enc)
        
        # DECODER
        embed_cate_col = embed_cate

        cat_list = list(embed_cate_col)
        cat_list.append(embed_interaction)
        
        embed_dec = torch.cat(cat_list, 2)
        embed_dec = self.dec_comb_proj(embed_dec)
        del cat_list

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        # print(f'embed_enc shape: {embed_enc.shape}')
        embed_dec = embed_dec.permute(1, 0, 2)
        # print(f'embed_dec shape: {embed_dec.shape}')
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, (self.cm.hidden_dim//2)*(self.cm.n_cate_cols))
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class GeneralizedLSTMConvATTN(nn.Module):
    def __init__(self, args):
        super(GeneralizedLSTMConvATTN, self).__init__()
        self.cm = CommonModule(args)
        self.n_cont_cols = self.cm.n_cont_cols
        self.n_cate_cols = self.cm.n_cate_cols
        self.args=args
        self.n_layers = self.cm.n_layers
        self.lstm_hidden_size = (self.cm.hidden_dim//2)*(self.cm.n_cate_cols+1)+((self.cm.hidden_dim//2)*self.n_cont_cols)
        # lstm layer hidden_dim size
        self.comb_proj = nn.Sequential(
                                    nn.Linear((self.cm.hidden_dim//2)*
                                    (self.cm.n_cate_cols+1), (self.cm.hidden_dim//2)*(self.cm.n_cate_cols+1)),
                                    # nn.LayerNorm(self.cm.hidden_dim)
                            )
        
        self.lstm = nn.LSTM(self.lstm_hidden_size,
                            self.lstm_hidden_size,
                            self.cm.n_layers,
                            batch_first=True)
        
        # 연속형 전용 layer(deprecated)
        # self.embedding_cont_cols = nn.Sequential(
        #     nn.Linear(self.n_cont_cols,(self.cm.hidden_dim//2)*self.n_cont_cols),
        #     nn.LayerNorm(self.cm.hidden_dim*self.n_cont_cols)
        # )

        # if args.model.lower() == 'lstmconvattn' :
        #     self.config = ConvBertConfig( 
        #     3, # not used
        #     hidden_size=self.hidden_dim,
        #     num_hidden_layers=1,
        #     num_attention_heads=self.n_heads,
        #     intermediate_size=self.hidden_dim,
        #     hidden_dropout_prob=self.drop_out,
        #     attention_probs_dropout_prob=self.drop_out,
        #     )
        #     self.attn = ConvBertEncoder(self.config)
        # elif args.model.lower() == 'lstmrobertaattn':
        #     self.config = RobertaConfig( 
        #     3, # not used
        #     hidden_size=self.hidden_dim,
        #     num_hidden_layers=1,
        #     num_attention_heads=self.n_heads,
        #     intermediate_size=self.hidden_dim,
        #     hidden_dropout_prob=self.drop_out,
        #     attention_probs_dropout_prob=self.drop_out,
        #     )
        #     self.attn = RobertaEncoder(self.config)
        # elif args.model.lower() == 'lstmalbertattn':
        #     self.config = AlbertConfig( 
        #     3, # not used
        #     hidden_size=self.hidden_dim,
        #     num_hidden_layers=1,
        #     num_attention_heads=self.n_heads,
        #     intermediate_size=self.hidden_dim,
        #     hidden_dropout_prob=self.drop_out,
        #     attention_probs_dropout_prob=self.drop_out,
        #     )
        #     # self.attn = AlbertAttention(self.config)
        #     # self.attn - AlbertModel(self.config)
        #     self.attn = AlbertTransformer(self.config)
        # else:
        #     self.config = BertConfig( 
        #     3, # not used
        #     hidden_size=self.hidden_dim,
        #     num_hidden_layers=1,
        #     num_attention_heads=self.n_heads,
        #     intermediate_size=self.hidden_dim,
        #     hidden_dropout_prob=self.drop_out,
        #     attention_probs_dropout_prob=self.drop_out,
        #     )
        #     self.attn = BertEncoder(self.config)

        self.config = BertConfig( 
            3, # not used
            hidden_size=self.lstm_hidden_size,
            num_hidden_layers=1,
            num_attention_heads=self.cm.n_heads,
            intermediate_size=self.cm.hidden_dim,
            hidden_dropout_prob=self.cm.drop_out,
            attention_probs_dropout_prob=self.cm.drop_out,
        )
        self.attn = BertEncoder(self.config)
    
        # Fully connected layer
        # self.fc = nn.Linear(self.n_embedding_hidden_dim*2, 1)
        self.fc = nn.Linear(self.lstm_hidden_size, self.args.max_seq_len)

        self.activation = self.cm.activation
        
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.cm.n_layers,
            batch_size,
            self.lstm_hidden_size
        )
            # self.cm.hidden_dim*2)
        h = h.to(self.cm.device)

        c = torch.zeros(
            self.cm.n_layers,
            batch_size,
            self.lstm_hidden_size
            )
            # self.cm.hidden_dim*2)
        c = c.to(self.cm.device)

        return (h, c)

    def forward(self, input):
        #common module 참고
        embed_cate,mask,embed_interaction,embed_cont,gather_index = self.cm.embed(input)

        batch_size = input[0].size(0)

        # 연속형 - 각자 다른 layer를 거친 후 concat
        # (batch, max_seq_len, 1)
        
        cat_list = [embed_interaction]
        cat_list.extend(embed_cate)
        
        embed = torch.cat(cat_list, 2)
        
        X = self.comb_proj(embed)
        
        if self.args.concat_reverse: # 연속형, 범주형 concat하는 순서 (True : 연속형, 범주형, False : 범주형, 연속형)
            if embed_cont is not None and X is not None:
                X = torch.cat([embed_cont,X],2)
            elif embed_cont is None:
                pass
        else:
            if embed_cont is not None and X is not None:
                X = torch.cat([X,embed_cont],2)
            elif embed_cont is None:
                pass
 
        hidden = self.init_hidden(batch_size)

        out, hidden = self.lstm(X, hidden)
        
        out = out.contiguous().view(batch_size, self.args.max_seq_len, -1)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:,-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)
        
        del hidden
        del input
        del extended_attention_mask
        del sequence_output
        del encoded_layers
        gc.collect()
        return preds