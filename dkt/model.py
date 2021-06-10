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
class CommonModule:
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
        self.embedding_cate = [
            nn.Embedding(v+1,self.hidden_dim//2)
            for k,v in self.args.cate_dict.items()
        ]
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
        # print(f'intercation : {interaction.shape}')
        return embed_cate,mask,embed_interaction,embed_cont,gather_index

    def future_mask(self,max_seq_len):
        future_mask = np.triu(np.ones((max_seq_len, max_seq_len)), k=1).astype('bool')
        return torch.from_numpy(future_mask)


'''
src = "https://www.kaggle.com/gannonreynolds/saint-riid-0-798"
'''
class FFN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.layer2(   self.relu(   self.layer1(x)))

class Encoder(nn.Module):
    def __init__(self, n_in, seq_len=100, embed_dim=128, nheads=4):
        super().__init__()
        self.seq_len = seq_len

        self.part_embed = nn.Embedding(10, embed_dim)
        
        self.e_embed = nn.Embedding(n_in, embed_dim)
        self.e_pos_embed = nn.Embedding(seq_len, embed_dim)
        self.e_norm = nn.LayerNorm(embed_dim)
        
        self.e_multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nheads, dropout=0.2)
        self.m_norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim)
    
    def forward(self, e, p, first_block=True):
        
        if first_block:
            e = self.e_embed(e)
            p = self.part_embed(p)
            e = e + p
         
        pos = torch.arange(self.seq_len).unsqueeze(0).to(device)
        e_pos = self.e_pos_embed(pos)
        e = e + e_pos
        e = self.e_norm(e)
        e = e.permute(1,0,2) #[bs, s_len, embed] => [s_len, bs, embed]     
        n = e.shape[0]
        
        att_mask = future_mask(n).to(device)
        att_out, _ = self.e_multi_att(e, e, e, attn_mask=att_mask)
        m = e + att_out
        m = m.permute(1,0,2)
        
        o = m + self.ffn(self.m_norm(m))
        
        return o

class GeneralizedSaintPlus(nn.Module):
    
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