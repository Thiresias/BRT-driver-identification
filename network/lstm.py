import torch.nn as nn
from torch.autograd import Variable
import torch
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
# from .mamba.mamba import Mamba, ModelArgs

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_size=56*3, output_size=56, hidden_size=768, LSTMnum_layer=1, batch_size=32, seq_length=39):
        super(LSTMClassifier, self).__init__()  # Correctly initialize the superclass
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=LSTMnum_layer, batch_first=False)
        self.fc1, self.fc2, self.fc3= nn.Linear(hidden_size+(embedding_size*(embedding_size-1))//2, output_size), nn.Linear(hidden_size//2, hidden_size//4), nn.Linear(hidden_size//4, output_size)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_size//2), nn.BatchNorm1d(hidden_size//4)
        self.sig = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layer = LSTMnum_layer
        self.TripletLoss = nn.TripletMarginLoss()
        # Need to define attention layer

    def forward(self, input, corr):
        input = input.permute(1,0,2)
        # The attention layer should happen before the lstm
        h_0 = Variable(torch.zeros(self.num_layer, input.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layer, input.shape[1], self.hidden_size).cuda())
        
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        h_dim = h_n[-1, :, :].shape
        # print(h_dim)
        hidden = h_n[-1, :, :].reshape(h_dim[0]//2, 2, h_dim[1])
        # triplet_loss = self.TripletLoss(hidden[:, 2, :], hidden[:, 0, :], hidden[:, 1, :])
        # anchor_pos = torch.sum(torch.norm(hidden[:, 2, :] - hidden[:, 0, :], dim=1))
        # Classifier
        # print(hidden.shape)
        # print(corr.shape)
        y = self.fc1(torch.cat((hidden.reshape(h_dim[0], h_dim[1]), corr), dim=1))
        # y = self.bn1(y)
        # y = self.fc2(y)
        # y = self.bn2(y)
        # y = self.fc3(y)

        return y
    

class LSTMClassifier_ldm(nn.Module):
    def __init__(self, embedding_size=68, output_size=2, hidden_size=32, LSTMnum_layer=1, batch_size=32, seq_length=40):
        super(LSTMClassifier_ldm, self).__init__()  # Correctly initialize the superclass
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=LSTMnum_layer, batch_first=False)
        self.fc1, self.fc2, self.fc3= nn.Linear(hidden_size, output_size), nn.Linear(hidden_size//2, hidden_size//4), nn.Linear(hidden_size//4, output_size)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_size//2), nn.BatchNorm1d(hidden_size//4)
        self.sig = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.numlayers = LSTMnum_layer
        self.firstconv = nn.Conv2d(2, 1, (1,1))

    def forward(self, input):
        input = self.firstconv(input.permute(0,3,1,2)).permute(2,0,3, 1).squeeze()
        # The attention layer should happen before the lstm
        h_0 = Variable(torch.zeros(self.numlayers, input.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.numlayers, input.shape[1], self.hidden_size).cuda())
        
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))

        # Classifier
        y = self.fc1(h_n[-1, :, :])
        # y = self.bn1(y)
        # y = self.fc2(y)
        # y = self.bn2(y)
        # y = self.fc3(y)

        return y
    

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_size=6, output_size=1, hidden_size=6, LSTMnum_layer=1, batch_size=32, seq_length=50):
        super(BiLSTMClassifier, self).__init__()  # Correctly initialize the superclass
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=LSTMnum_layer, batch_first=False, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn_lstm = nn.BatchNorm1d(2*hidden_size)
        self.bn_fc = nn.BatchNorm1d(hidden_size)
        self.sig = nn.Sigmoid()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        

    def forward(self, input):
        h_0 = Variable(torch.zeros(2, input.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(2, input.shape[1], self.hidden_size).cuda())
        
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        # Reshape output (D*num_layer, N, H_out) --> (N, D*Num_layer, H_out)
        h_n = h_n.permute(1,0,2)
        # Flatten the two last dimensions (N, D*Num_layer, H_out) --> (N, D*Num_layer*H_out)
        h_n = h_n.flatten(1)
        y = self.bn_lstm(h_n)
        y = self.fc1(y)
        y = self.bn_fc(y)
        y = self.fc2(y)
        score = self.sig(y)

        return score


class BiLSTMRecon(nn.Module):
    def __init__(self, embedding_size=6, output_size=10, hidden_size=16, LSTMnum_layer=6, batch_size=32):
        super(BiLSTMRecon, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=LSTMnum_layer)
        # Output size is set to reconstruct 10 samples, assuming each sample is represented by a single feature
        self.forward_reconstruct = nn.Linear(hidden_size, output_size)  # For forward mode reconstruction
        self.backward_reconstruct = nn.Linear(hidden_size, output_size)  # For backward mode reconstruction
        self.final_reconstruct = nn.Linear(2*output_size, output_size) # To reconstruct the signal with forward and backward
        self.bn = nn.BatchNorm1d(output_size) # To enhance generalization
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layer = LSTMnum_layer

    def extract_fwd_or_bwd_recon(self, input, mode='forward'):
        # Input shape is assumed to be (50, batch_size, input_size), where input_size is typically 1 for discrete signals
        if mode == 'forward':
            # Use the first 20 samples for forward mode
            input_processed = input[:20, :, :]
        elif mode == 'backward':
            # Use the last 20 samples for backward mode
            input_processed = input[-20:, :, :]
        else:
            raise ValueError("Mode must be 'forward' or 'backward'.")

        h_0 = Variable(torch.zeros(self.num_layer ,input_processed.size(1), self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layer ,input_processed.size(1), self.hidden_size).cuda())
        
        # Processing with LSTM
        lstm_out, (h_n, c_n) = self.lstm(input_processed, (h_0, c_0))
        
        # Selecting the appropriate linear layer for reconstruction based on mode
        if mode == 'forward':
            reconstructed =  self.forward_reconstruct(h_n[:, :, :])
        elif mode == 'backward':
            reconstructed = self.backward_reconstruct(h_n[:, :, :])

        return reconstructed
    
    def forward(self, x):
        y_forward = self.extract_fwd_or_bwd_recon(x, mode='forward')
        y_forward = self.bn(y_forward)
        y_backward = self.extract_fwd_or_bwd_recon(x, mode='backward')
        y_backward = self.bn(y_backward)

        # Concatenate forward and backward reconstructions
        y = torch.cat((y_forward, y_backward), dim=1)

        # Last layer: Reconstruct the signal with forward and backward information
        reconstructed_signal = self.final_reconstruct(y)

        return reconstructed_signal


class Attention(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(input_dim, attn_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (seq_length, batch, input_dim)
        scores = self.attn_weights(x)  # shape: (seq_length, batch, attn_dim)
        scores = scores.mean(dim=2)  # averaging attention scores over the dimension
        attn_scores = self.softmax(scores)  # shape: (seq_length, batch)
        attn_scores = attn_scores.unsqueeze(-1)  # shape: (seq_length, batch, 1)
        context = x * attn_scores  # shape: (seq_length, batch, input_dim)
        context_vector = context.sum(dim=0)  # shape: (batch, input_dim)
        return context_vector

class LSTMClassifier_wAttention(nn.Module):
    def __init__(self, embedding_size=56*3, output_size=56, hidden_size=128, LSTMnum_layer=1, batch_size=32, seq_length=39):
        super(LSTMClassifier_wAttention, self).__init__()
        self.attention = Attention(embedding_size, embedding_size)  # Assume attn_dim = input_dim for simplicity
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=LSTMnum_layer, batch_first=False)
        self.label = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=6, batch_first=False)

    def forward(self, input):
        attn_output, _ = self.multihead_attn(input, input, input)
        # Reshape context_vector to (seq_length, batch, input_dim) if needed
        h_0 = Variable(torch.zeros(1, attn_output.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, attn_output.shape[1], self.hidden_size).cuda())
        # Pass only the context_vector or combine it with original input as needed
        output, (h_n, c_n) = self.lstm(attn_output, (h_0, c_0))
        y = self.label(h_n[-1, :, :])
        score = self.sig(y)

        return score






# --- Self supervised --- #


# Inspired from https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series/tree/main
class Encoder(nn.Module):
    def __init__(self, seq_len=50, n_features=6, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'ENCODER input dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.rnn1(x)
        # print(f'ENCODER output rnn1 dim: {x.shape}')
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'ENCODER output rnn2 dim: {x.shape}')
        # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
        return hidden_n.reshape((batch_size, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len=50, input_dim=64, n_features=6, batch_size=32):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'DECODER input dim: {x.shape}')
        x = x.repeat(1, self.seq_len) # todo testare se funziona con più feature
        # print(f'DECODER repeat dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)


######
# MAIN
######

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len=50, n_features=6, embedding_dim=64, device='cuda', batch_size=32):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ViT
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224').cuda()

def encode_image_with_vit(image):
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    # Move the inputs to the GPU
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Get the image encoding from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state of the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].squeeze()

    return cls_embeddings


class LSTMClassifier_ViT(nn.Module):
    def __init__(self, embedding_size=56*3, output_size=56, hidden_size=768, LSTMnum_layer=1, batch_size=32, seq_length=39):
        super(LSTMClassifier_ViT, self).__init__()  # Correctly initialize the superclass
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=LSTMnum_layer, batch_first=False)
        self.fc1, self.fc2, self.fc3= nn.Linear(hidden_size, output_size), nn.Linear(hidden_size//2, hidden_size//4), nn.Linear(hidden_size//4, output_size)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_size//2), nn.BatchNorm1d(hidden_size//4)
        self.sig = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layer = LSTMnum_layer
        self.TripletLoss = nn.TripletMarginLoss()
        # Need to define attention layer

    def forward(self, input):
        print(input.shape)
        lstm_input = list()
        for batch in input:
            # Encode the cropped head batch (batch dim: (T, N, M, C))
            embeddings = encode_image_with_vit(batch)

            embeddings = embeddings - torch.mean(embeddings, dim=0)
            # embeddings dim: (T, H)
            lstm_input.append(embeddings) # lstm_input dim: (b_s x 3, T, H)
        lstm_input = torch.stack(lstm_input).permute(1,0,2) # dim: (T, b_s x 2, H)

        # The attention layer should happen before the lstm
        h_0 = Variable(torch.zeros(self.num_layer, lstm_input.shape[1], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layer, lstm_input.shape[1], self.hidden_size).cuda())
        
        output, (h_n, c_n) = self.lstm(lstm_input, (h_0, c_0))
        h_dim = h_n[-1, :, :].shape
        hidden = h_n[-1, :, :].reshape(h_dim[0]//2, 2, h_dim[1])
        # triplet_loss = self.TripletLoss(hidden[:, 2, :], hidden[:, 0, :], hidden[:, 1, :])
        # anchor_pos = torch.sum(torch.norm(hidden[:, 2, :] - hidden[:, 0, :], dim=1))
        # Classifier
        y = self.fc1(h_n[-1,:, :])
        # y = self.bn1(y)
        # y = self.fc2(y)
        # y = self.bn2(y)
        # y = self.fc3(y)

        return y, hidden


class S4Layer(nn.Module):
    def __init__(self, d_model, state_dim):
        super(S4Layer, self).__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, d_model))
        self.C = nn.Parameter(torch.randn(d_model, state_dim))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        state = torch.zeros(batch_size, self.state_dim).to(x.device)
        outputs = []
        for t in range(seq_len):
            state = torch.nn.functional.relu(self.A @ state.transpose(0, 1) + self.B @ x[:, t].transpose(0, 1)).transpose(0, 1)
            output = self.C @ state.transpose(0, 1)
            outputs.append(output.transpose(0, 1))
        return torch.stack(outputs, dim=1)

class S4Classifier(nn.Module):
    def __init__(self, d_model, state_dim, num_classes):
        super(S4Classifier, self).__init__()
        self.s4_layer = S4Layer(d_model, state_dim)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.s4_layer(x)
        x = x[:, -1, :]  # Use the last output of the sequence
        x = self.fc(x)
        return x


class MAMBAClassifier(nn.Module):
    def __init__(self, d_model, state_dim, num_classes):
        super(MAMBAClassifier, self).__init__()
        args = ModelArgs(d_model= d_model, n_layer= 6, input_size= 12)
        self.mamba = Mamba(args).to('cuda')
        self.fc = nn.Linear(d_model, num_classes).to('cuda')
        self.fc_input = nn.Linear(12, d_model).to('cuda')
        for param in self.mamba.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.fc_input(x)
        x = self.mamba(x)
        x = x[:, -1, :]  # Use the last output of the sequence
        x = self.fc(x)
        return x