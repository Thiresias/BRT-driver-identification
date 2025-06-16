import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .generator import Generator


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


class BRTDFD(nn.Module):
    def __init__(self, embedding_size=56*3, output_size=56, hidden_size=512, nhead=10, num_encoder_layers=3, batch_size=32, seq_length=39):
        super(BRTDFD, self).__init__()
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
        lia = lia - torch.mean(lia, dim=0) # REMOVE COMMENT FOR BRT-DFD+
        lia = lia / torch.std(lia, dim=0) # BRT-DFD ++ ?
        
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

        return y, transformer_output
    
    
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        """
        ArcFace loss implementation
        :param embedding_size: Dimension of the feature embeddings
        :param num_classes: Number of classes
        :param margin: Angular margin (default: 0.5)
        :param scale: Feature scaling factor (default: 64.0)
        """
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        
        # Weight matrix (fully connected layer)
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Forward pass for ArcFace loss
        :param embeddings: (batch_size, embedding_size) tensor
        :param labels: Ground-truth labels (batch_size,)
        """
        # Normalize embeddings and weight vectors
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = torch.matmul(embeddings, weight_norm.T)  # Shape: (batch_size, num_classes)

        # Add margin to the correct class
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        theta_m = theta + self.margin
        cosine_m = torch.cos(theta_m)

        # Create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply the margin to the correct class logits
        logits = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)

        # Scale logits
        logits *= self.scale

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss