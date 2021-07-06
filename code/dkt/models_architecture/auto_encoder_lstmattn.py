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


class AutoEncoderLSTMATTN(nn.Module):
    def __init__(self, args):
        super(AutoEncoderLSTMATTN,self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        #dev
        self.n_other_features = self.args.n_other_features
        print(f'other features cont : {self.n_other_features}')

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        # other feature
        self.f_cnt = len(self.n_other_features) # feature의 개수
        self.embedding_other_features = [nn.Embedding(self.n_other_features[i]+1, self.hidden_dim//3) for i in range(self.f_cnt)]

        self.comb_proj = nn.Linear((self.hidden_dim//3)*(4+self.f_cnt), self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        # self.embedding_test = nn.Embedding(100,self.hidden_dim//3)
        if args.model.lower() == 'lstmconvattn' :
            self.config = ConvBertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
            )
            self.attn = ConvBertEncoder(self.config)
        elif args.model.lower() == 'lstmrobertaattn':
            self.config = RobertaConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
            )
            self.attn = RobertaEncoder(self.config)
        elif args.model.lower() == 'lstmalbertattn':
            self.config = AlbertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
            )
            # self.attn = AlbertAttention(self.config)
            # self.attn - AlbertModel(self.config)
            self.attn = AlbertTransformer(self.config)
        else:
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
        # print(f'input 길이 : {len(input)}')
        
        # input의 순서는 test, question, tag, _, mask, interaction, (...other features), gather_index(안 씀)

        # for i,e in enumerate(input):
        #     print(f'i 번째 : {e[i].shape}')
        test = input[0]
        question = input[1]
        tag = input[2]

        mask = input[4]
        interaction = input[5]
        
        other_features = [input[i] for i in range(6,len(input)-1)]

        batch_size = interaction.size(0)
        
        # Embedding
        # print(interaction.shape)
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        # dev
        embed_other_features =[] 
        
        for i,e in enumerate(self.embedding_other_features):
            # print(f'{i}번째 : {e}')
            # print(f'최댓값(전) : {torch.max(other_features[i])}')
            # print(f'최솟값(전) : {torch.min(other_features[i])}')
            embed_other_features.append(e(other_features[i]))
            # print(f'최댓값(후) : {torch.max(other_features[i])}')
            # print(f'최솟값(후) : {torch.min(other_features[i])}')
        
        cat_list = [embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           ]
        cat_list.extend(embed_other_features)
        embed = torch.cat(cat_list, 2)
        

        X = self.comb_proj(embed)

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