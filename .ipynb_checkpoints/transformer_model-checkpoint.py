# transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_size=3, output_size=1, embed_size=32, hidden_size=64, e_num_layers=1, d_num_layers=1, num_heads=4, dropout_prob=0.1, device="cpu"):
        super(TransformerModel, self).__init__()
        self.device = device
        self.embedding = nn.Linear(input_size, embed_size).to(self.device)
        self.dec_embedding = nn.Linear(output_size, embed_size).to(self.device)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout_prob,
                batch_first=True
            ),
            num_layers=e_num_layers
        ).to(self.device)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout_prob,
                batch_first=True
            ),
            num_layers=d_num_layers
        ).to(self.device)
        self.output_layer = nn.Linear(embed_size, output_size).to(self.device)

    def forward(self, X, dec_input):
        X = self.embedding(X.to(self.device))
        encoder_output = self.encoder(X)
        dec_input = dec_input.unsqueeze(-1)
        dec_input = self.dec_embedding(dec_input.to(self.device))
        tgt_mask = torch.triu(torch.ones(dec_input.size(1), dec_input.size(1)), diagonal=1).bool().to(self.device)
        decoder_output = self.decoder(dec_input, encoder_output, tgt_mask=tgt_mask)
        output = self.output_layer(decoder_output)
        return output

    def training_step(self, X, y, optimizer):
        '''Training step with teacher forcing for autoregressive prediction'''
        
        X, y = X.to(self.device), y.to(self.device)
        X_embedded = self.embedding(X)
        encoder_output = self.encoder(X_embedded)
        
        dec_input = y[:, 0].unsqueeze(1).unsqueeze(-1)
        output_seq = []

        for t in range(y.shape[1]):
            dec_input_embedded = self.dec_embedding(dec_input)
            decoder_output = self.decoder(dec_input_embedded, encoder_output)
            voltage_prediction = self.output_layer(decoder_output[:, -1, :])
            output_seq.append(voltage_prediction)
            if t < y.shape[1] - 1:
                dec_input = y[:, t + 1].unsqueeze(1).unsqueeze(-1)
        
        output_seq = torch.cat(output_seq, dim=1).unsqueeze(-1)
        y = y.unsqueeze(-1)
        
        loss = F.mse_loss(output_seq, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        return loss.item()

    def validation_step(self, X, y):
        '''Validation step with autoregressive prediction'''
        
        # Move input and target tensors to the specified device
        X, y = X.to(self.device), y.to(self.device)
        
        # Pass input sequence through embedding layer and encoder
        X_embedded = self.embedding(X)
        encoder_output = self.encoder(X_embedded)
        
        # Initialize decoder input with the first target value (for autoregressive prediction)
        dec_input = y[:, 0].unsqueeze(1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
        output_seq = []
    
        # Autoregressive prediction for each timestep in the target sequence
        for t in range(y.shape[1]):
            # Embed the decoder input at the current timestep
            dec_input_embedded = self.dec_embedding(dec_input)
            
            # Decoder pass with cross-attention to encoder output
            decoder_output = self.decoder(dec_input_embedded, encoder_output)
            
            # Predict voltage at the next timestep
            voltage_prediction = self.output_layer(decoder_output[:, -1, :])
            # Ensure voltage_prediction has shape [batch_size, 1, 1]
            voltage_prediction = voltage_prediction.unsqueeze(1)  # Add extra dimension
            
            # Collect prediction and set it as the next decoder input
            output_seq.append(voltage_prediction)
            dec_input = voltage_prediction  # Use model's own prediction as input for next timestep
        
        # Concatenate all timestep predictions to form the output sequence
        output_seq = torch.cat(output_seq, dim=1)  # Shape: [batch_size, seq_len]
        y = y.unsqueeze(-1)  # Make target shape compatible with output_seq
        
        # Calculate the mean squared error loss
        loss = F.mse_loss(output_seq, y)
        
        return loss.item()

