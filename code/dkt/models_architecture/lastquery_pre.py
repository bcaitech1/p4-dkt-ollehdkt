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


import re

##### PrePadding

class Feed_Forward_block_Pre(nn.Module):

    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery_Pre(nn.Module):
    def __init__(self, args):
        super(LastQuery_Pre, self).__init__()

        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        #userID때문에 하나 뺌
        cate_len=len(args.cate_feats)-1
        #answerCode 때문에 하나 뺌
        cont_len=len(args.cont_feats)-1

        # Embedding 
        # cate Embedding 
        self.cate_embedding_list = nn.ModuleList([nn.Embedding(max_val+1, (self.hidden_dim//2)//cate_len) for max_val in list(args.cate_feat_dict.values())[1:]]) 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, (self.hidden_dim//2)//cate_len)

        # cont Embedding
        self.cont_embedding = nn.Linear(1, (self.hidden_dim//2)//cont_len)


        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        
        # comb linear
        self.cate_comb_proj = nn.Linear(((self.hidden_dim//2)//cate_len)*(cate_len+1), self.hidden_dim//2) #interaction을 나중에 더하므로 +1
        self.cont_comb_proj = nn.Linear(((self.hidden_dim//2)//cont_len)*cont_len, self.hidden_dim//2)

        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block_Pre(self.hidden_dim)      


        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


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

        # cate Embedding
        cate_feats_embed=[]
        embed_interaction = self.embedding_interaction(interaction)
        cate_feats_embed.append(embed_interaction)

        for i, cate_feat in enumerate(cate_feats): 
            cate_feats_embed.append(self.cate_embedding_list[i](cate_feat))
        
        # unsqueeze cont feats shape & embedding
        cont_feats_embed=[]
        for cont_feat in cont_feats:
            cont_feat=cont_feat.unsqueeze(-1)
            cont_feats_embed.append(self.cont_embedding(cont_feat))
            
        
        #concat cate, cont feats
        embed_cate = torch.cat(cate_feats_embed, 2)
        embed_cate=self.cate_comb_proj(embed_cate)

        embed_cont = torch.cat(cont_feats_embed, 2)
        embed_cont=self.cont_comb_proj(embed_cont)


        embed = torch.cat([embed_cate,embed_cont], 2)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
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
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)
        # print(preds)

        return preds