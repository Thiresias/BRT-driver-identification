import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from .generator import Generator
import os
from transformers import BertModel, BertTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_size=56*3, output_size=56, hidden_size=512, nhead=10, num_encoder_layers=3, batch_size=32, seq_length=39):
        super(TransformerClassifier, self).__init__()
        self.embedding_size = embedding_size
        
        # Transformer encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.pos_encoder = PositionalEncoding(embedding_size, max_len=41)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Classification head
        self.fc1, self.fc2, self.fc3 = nn.Linear(embedding_size, embedding_size // 2), nn.Linear(embedding_size // 2, embedding_size // 4), nn.Linear(embedding_size // 4, output_size)
        self.bn1 = nn.BatchNorm1d(embedding_size // 2)
        self.bn2 = nn.BatchNorm1d(embedding_size // 4)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(0.1)

        # Hyperparameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # Backbone
        self.backbone = Generator(256, 512, 20, 1).cuda()
        print('Path to model found') if os.path.exists(r'LIA_encoder/checkpoints/vox.pt') else print('Path to model NOT found !')
        weight = torch.load(r'LIA_encoder/checkpoints/vox.pt', map_location=lambda storage, loc: storage)['gen']
        self.backbone.load_state_dict(weight)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input):
        mvt_feature = []
        for sample in input:
            mvt_feature.append(self.backbone(sample[0, :, :, :].unsqueeze(0), sample, sample[0, :, :, :].unsqueeze(0)))
            
        lia = torch.stack(mvt_feature)

        lia = lia.permute(1, 0, 2)  # (seq_length, batch_size, embedding_size)
        # # Center lia output
        # lia = lia - torch.mean(lia, dim = 0) # REMOVE COMMENT FOR BRT-DFD+
        
        cls_tokens = self.cls_token.expand(1, lia.shape[1], lia.shape[2])
        lia = torch.cat((cls_tokens, lia), dim=0)
        lia = self.pos_encoder(lia)
        
        
        transformer_output = self.transformer_encoder(lia)
        transformer_output = transformer_output[0,:,:] # [CLS] token output

        ## Classifier
        # 1st layer
        y = self.fc1(transformer_output)
        y = self.bn1(y)
        y = self.dropout(y)
        # 2nd layer
        y = self.fc2(y)
        y = self.bn2(y)
        y = self.dropout(y)
        # 3rd layer
        y = self.fc3(y)
        y = self.bn3(y)
        y = self.dropout(y)

        return y, lia
    

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_size=56*3, output_size=56, hidden_size=512, LSTMnum_layer=1, batch_size=32, seq_length=39):
        super(LSTMClassifier, self).__init__()  # Correctly initialize the superclass
        self.fc_selec = nn.Linear(512, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=LSTMnum_layer)
        # self.fc1, self.fc2, self.fc3= nn.Linear(hidden_size, output_size), nn.Linear(hidden_size//2, hidden_size//4), nn.Linear(hidden_size//4, output_size)
        # self.bn1, self.bn2 = nn.BatchNorm1d(hidden_size//2), nn.BatchNorm1d(hidden_size//4)
        self.fc1, self.fc2, self.fc3 = nn.Linear(hidden_size, hidden_size // 2), nn.Linear(hidden_size // 2, hidden_size // 4), nn.Linear(hidden_size // 4, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(0.1)
        self.sig = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.backbone = Generator(256, 512, 20, 1).cuda()
        print('Path to model found') if os.path.exists(r'LIA_encoder/checkpoints/vox.pt') else print('Path to model NOT found !')
        weight = torch.load(r'LIA_encoder/checkpoints/vox.pt', map_location=lambda storage, loc: storage)['gen']
        self.backbone.load_state_dict(weight)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input):
        # print(input.shape)
        mvt_feature = []
        for sample in input:
            mvt_feature.append(self.backbone(sample[0, :, :, :].unsqueeze(0), sample, sample[0, :, :, :].unsqueeze(0)))
        lia = torch.stack(mvt_feature)
        # lia_diff = lia [:, :-1, :] - lia[:, 1:, :]
        # x = self.fc_selec(lia)
        h_0 = Variable(torch.zeros(1,input.shape[0], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1,input.shape[0], self.hidden_size).cuda())
        
        output, (h_n, c_n) = self.lstm(lia.permute(1,0,2), (h_0, c_0))
        # print(h_n[-1, :, :].shape)
        ## Classifier
        # 1st layer
        y = self.fc1(h_n[-1, :, :])
        y = self.bn1(y)
        y = self.dropout(y)
        # 2nd layer
        y = self.fc2(y)
        y = self.bn2(y)
        y = self.dropout(y)
        # 3rd layer
        y = self.fc3(y)
        y = self.bn3(y)
        y = self.dropout(y)

        return y, lia