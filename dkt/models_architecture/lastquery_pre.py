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
        self.cont_cols=self.args.cont_cols
        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)


        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        # self.n_other_features = self.args.n_other_features
        # print(self.n_other_features)
        self.cont_proj=nn.Linear(self.cont_cols,self.hidden_dim//2)

        # encoder combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim//2)

        # # other feature
        # self.f_cnt = len(self.n_other_features) # feature의 개수
        # self.embedding_other_features = [nn.Embedding(self.n_other_features[i]+1, self.hidden_dim//3) for i in range(self.f_cnt)]
        
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
        test, question,tag, correct, mask, interaction, solve_time, gather_index=input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        solve_time=solve_time.unsqueeze(-1) #shape(B,MSL) -> shape(B, MSL, 1)
        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_cont=self.cont_proj(solve_time)
        # dev
  
        # for i,e in enumerate(self.embedding_other_features):
        #     # print(f'{i}번째 : {e}')
        #     # print(f'최댓값(전) : {torch.max(other_features[i])}')
        #     # print(f'최솟값(전) : {torch.min(other_features[i])}')
        #     embed_other_features.append(e(other_features[i]))
        #     # print(f'최댓값(후) : {torch.max(other_features[i])}')
        #     # print(f'최솟값(후) : {torch.min(other_features[i])}')

        cat_list = [embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           ]
        # cat_list.extend(embed_other_features)

        embed = torch.cat(cat_list, 2)


        embed = self.comb_proj(embed)
        embed=torch.cat([embed, embed_cont], 2) #(batch,msl, 128)

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