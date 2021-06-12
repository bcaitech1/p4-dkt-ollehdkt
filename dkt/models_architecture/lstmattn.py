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

class LSTMATTN(nn.Module):
    
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        #userID때문에 하나 뺌
        cate_len=len(args.cate_feats)-1
        #answerCode 때문에 하나 뺌
        cont_len=len(args.cont_feats)-1
        # cate Embedding 
        self.cate_embedding_list = nn.ModuleList([nn.Embedding(max_val+1, (self.hidden_dim//2)//cate_len) for max_val in list(args.cate_feat_dict.values())[1:]]) 
        # cont Embedding
        self.cont_embedding = nn.Linear(1, (self.hidden_dim//2)//cont_len)

        # comb linear
        self.cate_comb_proj = nn.Linear(((self.hidden_dim//2)//cate_len)*(cate_len+1), self.hidden_dim//2) #interaction을 나중에 더하므로 +1
        self.cont_comb_proj = nn.Linear(((self.hidden_dim//2)//cont_len)*cont_len, self.hidden_dim//2)

        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, (self.hidden_dim//2)//cate_len)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
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
        
        X = torch.cat([embed_cate,embed_cont], 2)
        # print("cate와 cont를 concat한 shape : ", X.shape)
        
        hidden = self.init_hidden(batch_size)
        # print(f'{hidden[0].shape}, {hidden[1].shape}')
        out, hidden = self.lstm(X, hidden)
        # print(out.shape)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # print(out.shape)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds