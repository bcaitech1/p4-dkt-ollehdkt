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

class TfixupSaint(nn.Module):
    
    def __init__(self, args,Tfixup=True):
        super(TfixupSaint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.dropout
        self.dropout = 0.
        
        ### Embedding 
        # ENCODER embedding

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        self.n_other_features = self.args.n_other_features
        print(self.n_other_features)
        
        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//3)*(3+len(self.n_other_features)), self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//3)*(4+len(self.n_other_features)), self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)

        # # other feature
        self.f_cnt = len(self.n_other_features) # feature의 개수
        self.embedding_other_features = [nn.Embedding(self.n_other_features[i]+1, self.hidden_dim//3) for i in range(self.f_cnt)]
        

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

    # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixupbb Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0
        print(self.named_parameters)
        for name, param in self.named_parameters():
            print(f'name : {name}')
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*Norm.*', name) or re.match(r'.*norm*.*',name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}
        
        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param   
            elif re.match(r'.*Norm.*', name) or re.match(r'.*norm*.*',name):
                continue
            elif re.match(r'encoder.*dense.*weight$|encoder.*attention.output.*weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]
                
        self.load_state_dict(temp_state_dict)
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        # test, question, tag, _, mask, interaction, _ = input

        # # print(f'input 길이 : {len(input)}')
        
        # # input의 순서는 test, question, tag, _, mask, interaction, (...other features), gather_index(안 씀)

        # # for i,e in enumerate(input):
        # #     print(f'i 번째 : {e[i].shape}')
        test = input[0]
        question = input[1]
        tag = input[2]

        mask = input[4]
        interaction = input[5]
        
        other_features = [input[i] for i in range(6,len(input)-1)]
        
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        # # dev
        embed_other_features =[] 
        
        for i,e in enumerate(self.embedding_other_features):
            # print(f'{i}번째 : {e}')
            # print(f'최댓값(전) : {torch.max(other_features[i])}')
            # print(f'최솟값(전) : {torch.min(other_features[i])}')
            embed_other_features.append(e(other_features[i]))
            # print(f'최댓값(후) : {torch.max(other_features[i])}')
            # print(f'최솟값(후) : {torch.min(other_features[i])}')
        
        cat_list = [
            # embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           ]
        cat_list.extend(embed_other_features)
        embed_enc = torch.cat(cat_list, 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # DECODER     
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)

        cat_list = [
                    
                           embed_test,
                           embed_question,
                           embed_tag,
                        embed_interaction,
                           ]
        cat_list.extend(embed_other_features)
        embed_dec = torch.cat(cat_list, 2)

        embed_dec = self.dec_comb_proj(embed_dec)

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
        embed_dec = embed_dec.permute(1, 0, 2)
        
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