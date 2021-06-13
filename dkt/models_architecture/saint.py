from operator import index
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


class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.drop_out
        self.dropout =args.drop_out

        #userID때문에 하나 뺌
        cate_len=len(args.cate_feats)-1
        #answerCode 때문에 하나 뺌
        cont_len=len(args.cont_feats)-1

        ### Embedding 
        # ENCODER embedding - for cate
        # cate Embedding 
        self.cate_embedding_list = nn.ModuleList([nn.Embedding(max_val+1, (self.hidden_dim)//cate_len) for max_val in list(args.cate_feat_dict.values())[1:]]) 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, (self.hidden_dim)//cate_len)

        
        # DECODER embedding - for cont
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        # cont Embedding
        self.cont_embedding = nn.Linear(1, (self.hidden_dim)//cont_len)

        # comb linear
        self.cate_comb_proj = nn.Linear(((self.hidden_dim)//cate_len)*(cate_len+1), self.hidden_dim) #interaction을 나중에 더하므로 +1
        self.cont_comb_proj = nn.Sequential(
            nn.Linear(((self.hidden_dim)//cont_len)*cont_len, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)

        # # other feature
        # self.f_cnt = len(self.n_other_features) # feature의 개수
        # self.embedding_other_features = [nn.Embedding(self.n_other_features[i]+1, self.hidden_dim//3) for i in range(self.f_cnt)]
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        #userID가 빠졌으므로 -1
        cate_feats=input[:len(self.args.cate_feats)-1]
        # print("cate_feats개수",len(cate_feats))
        
        #answercode가 없으므로 -1
        cont_feats=input[len(self.args.cate_feats)-1:-4]
        # print("cont_feats개수",len(cont_feats))      
        interaction=input[-4]
        mask=input[-3]
        gather_index=input[-2]

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)


        # 신나는 embedding
        # ENCODER
        # cate Embedding
        cate_feats_embed=[]
        embed_interaction = self.embedding_interaction(interaction)
        cate_feats_embed.append(embed_interaction)

        for i, cate_feat in enumerate(cate_feats): 
            cate_feats_embed.append(self.cate_embedding_list[i](cate_feat))
        
        #concat cate for Encoder
        embed_cate = torch.cat(cate_feats_embed, 2)
        embed_enc=self.cate_comb_proj(embed_cate)

 
        # DECODER 
        # # unsqueeze cont feats shape & embedding
        cont_feats_embed=[]
        for cont_feat in cont_feats:
            cont_feat=cont_feat.unsqueeze(-1)
            cont_feats_embed.append(self.cont_embedding(cont_feat))
                
        embed_cont = torch.cat(cont_feats_embed, 2)
        embed_dec=self.cont_comb_proj(embed_cont)
     
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
        embed_dec = embed_dec.permute(1, 0, 2)#shape(batch,msl,hidden_dim) -> shape(msl,batch,hidden_dim)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)


        return preds