# =======================================================
# Name: Hamers Robin
# GitHub: Rhodham96
# Year: 2025
# Description: Attention is all you need - Build a GPT from scratch, helped with Andrej Kartpathy video "Let's build GPT: from scratch, in code, spelled out"
# =======================================================

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import string
#from sklearn.model_selection import train_test_split
import torch.optim as optim

# hyperparameters
dropout_rate = 0.1
vocab_size = 8000
#max_len = 50 # max seq len
n_embd = 384

file_path = "/Users/robinhamers/Downloads/kaggle.json"

with open(file_path, "r") as f:
    contenu = f.read()
    print(contenu)

# Step 1: Load the CSV file
df_import = pd.read_csv('eng_-french.csv')
print(df_import.head())

df = pd.DataFrame()
df['english'] = df_import['English words/sentences']
df['french'] = df_import['French words/sentences']
print(df.head())

# No need if file already created
# Create a combined file with English and French sentences for SentencePiece training
with open("combined_data.txt", "w") as file:
    for e, f in zip(df['english'], df['french']):
        file.write(e + '\n' + f + '\n')  # Add each English-French sentence pair.

# Train the SentencePiece model (bpe-based)
spm.SentencePieceTrainer.train(input='combined_data.txt', model_prefix='spm_model', vocab_size=vocab_size, model_type='bpe')

sp_en = spm.SentencePieceProcessor(model_file='spm_model.model')
sp_fr = spm.SentencePieceProcessor(model_file='spm_model.model')

def tokenize(text, sp_processor):
    return sp_processor.encode(text, out_type=str)  # Encode to subword tokens

# Tokenize English and French sentences
df['english_tokens'] = df['english'].apply(lambda x: tokenize(x, sp_en))
df['french_tokens'] = df['french'].apply(lambda x: tokenize(x, sp_fr))

def tokens_to_indices(tokens, sp_processor):
    return sp_processor.encode(' '.join(tokens), out_type=int)  # Convert to indices

df['english_indices'] = df['english_tokens'].apply(lambda x: tokens_to_indices(x, sp_en))
df['french_indices'] = df['french_tokens'].apply(lambda x: tokens_to_indices(x, sp_fr))

max_len = max(df['english_indices'].apply(len).max(), df['french_indices'].apply(len).max())

# Fonction pour ajouter du padding
def pad_sequence(seq, max_len, sp_model):
    pad_id = sp_model.piece_to_id('<pad>')  # Obtenir l'ID du token PAD
    return seq + [pad_id] * (max_len - len(seq))  # Ajouter le padding jusqu'à max_len

# Calculer la longueur maximale des phrases dans les deux langues
max_len = max(
    df['english_indices'].apply(len).max(),  # Longueur maximale pour l'anglais
    df['french_indices'].apply(len).max()   # Longueur maximale pour le français
)

# Dataset pour la traduction
class TranslationDataset(Dataset):
    def __init__(self, english_sentences, french_sentences, sp_en, sp_fr, max_len):
        # Tokenisation des phrases en anglais et français
        self.english_sentences = [sp_en.encode(sent, out_type=int) for sent in english_sentences]
        self.french_sentences = [sp_fr.encode(sent, out_type=int) for sent in french_sentences]
        self.max_len = max_len

        # Padding des séquences
        self.english_sentences = [pad_sequence(sent, max_len, sp_en) for sent in self.english_sentences]
        self.french_sentences = [pad_sequence(sent, max_len, sp_fr) for sent in self.french_sentences]

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.english_sentences[idx]), torch.tensor(self.french_sentences[idx])

# Exemple : Charger les phrases depuis ton DataFrame
english_sentences = df['english'].tolist()
french_sentences = df['french'].tolist()

print(f"max len = {max_len}")

# Créer le dataset
dataset = TranslationDataset(english_sentences, french_sentences, sp_en, sp_fr, max_len)

# Créer un DataLoader pour charger les données en lots
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Head(nn.Module):
    """ One head of self-attention (for encoder/decoder) """

    def __init__(self, head_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, encoder_out=None):
        """
        Arguments:
            x: Input tensor.
            mask: Optional mask for attention.
            encoder_out: Optional encoder output for cross-attention.
        """
        B, T, C = x.shape  # Get dimensions of the input tensor
        #print(f"x shape = {x.shape}")
        # If encoder_out is provided, use it for keys and values (cross-attention)
        encoder_seq_len = []
        if encoder_out is not None:
            # Project encoder_out to embed_dim
            #print(f"encoder out = {encoder_out.shape}")
            _, encoder_seq_len, encoder_dim = encoder_out.shape  # Get encoder_out dimensions
            #print(f"encoder dim = {encoder_dim}")
            self.encoder_proj = nn.Linear(encoder_dim, self.embed_dim)  # Initialize encoder_proj
            encoder_out = self.encoder_proj(encoder_out)  # Project to embed_dim

            # Calculate keys and values from encoder_out
            k = self.key(encoder_out)  # (B, T, head_size)
            v = self.value(encoder_out)  # (B, T, head_size)
        else:  # Otherwise, use x for keys and values (self-attention)
            k = self.key(x)  # (B, T, head_size)
            v = self.value(x)  # (B, T, head_size)

        # Calculate query from input x
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores
        wei = torch.bmm(q, k.transpose(1, 2)) * (C ** -0.5)  # (B, T, T)
        #print("Attention weights (wei):", wei)
        # Apply optional padding mask
        if mask is not None:
            # Reshape mask to match wei's shape for self-attention or cross-attention
            # Assuming mask shape is (batch_size, 1, target_sequence_length, source_sequence_length)

            # Apply mask to attention scores
            if encoder_seq_len:
                # If the mask has a dimension of size 256 in the third axis, we need to reshape it
                if mask.shape[2] == 256:
                    mask = mask[:, :, :max_len-1]  # Slice the mask to have shape [batch_size, seq_len, seq_len]
                    
                #mask = mask.unsqueeze(1)  # Shape: [batch_size, 1, seq_len]
                
                mask = mask.expand(-1, encoder_seq_len, encoder_seq_len)
                
            wei = wei.masked_fill(mask == 0, float('-inf'))  # Apply mask
            

        # Apply softmax to get attention weights
        wei = F.softmax(wei, dim=-1)
        

        # Apply dropout
        #wei = self.dropout(wei)

        # Expand v to match the sequence length T
        v = v.expand(-1, T, -1)  # Expands the second dimension (sequence length)
        #print(f"wei shape = {wei.shape}")
        #print(f"v shape = {v.shape}")
        
        out = wei @ v  # (B, T, head_size)
        #print(f"out shape = {out.shape}")
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multi-head attention mechanism """

    def __init__(self, embed_dim, num_heads, head_size, dropout=dropout_rate):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([Head(head_size, embed_dim=embed_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embed_dim)  # Projection layer with correct input/output dimensions
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, encoder_out=None):
        # Apply each head to the input and concatenate the results
        
        out = torch.cat([h(x, mask, encoder_out) for h in self.heads], dim=-1)

        # Project the concatenated outputs to the original embedding dimension
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non linearity"""

    def __init__(self, embd_dim, ff_dim, dropout=dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_dim, 4*ff_dim),
            nn.ReLU(),
            nn.Linear(4*ff_dim, embd_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
# Transformer Encoder Layer

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=dropout_rate):
        super().__init__()
        head_size = embed_dim // num_heads
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, head_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Self-Attention + Add & Norm
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + attn_out)

        # Feedforward + Add & Norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        #print(x.shape)
        return self.dropout(x)
    
# Transformer Decoder Layer

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=dropout_rate):
        super().__init__()
        head_size = embed_dim // num_heads # Calculate head_size here
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, head_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, head_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):

        # Self-Attention + Add & Norm
        self_attn_out = self.self_attn(x, tgt_mask)
        x = self.norm1(x + self_attn_out)
        x = self.dropout(x)
        # Cross-Attention (Encoder-Decoder)
        src_mask = src_mask.expand(-1, max_len-1, -1)
        cross_attn_out = self.cross_attn(x, encoder_out, src_mask)
        x = self.norm2(x + cross_attn_out)
        x = self.dropout(x)
        # Feedforward + Add & Norm
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return self.dropout(x)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return x
    
# Full Transformer

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout=dropout_rate):
      super().__init__()
      self.embedding = nn.Embedding(vocab_size, embed_dim)
      self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout=dropout)
      self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, dropout=dropout)
      self.fc_out = nn.Linear(embed_dim, vocab_size)
      # Dropout layer
      self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.embedding(src))
        tgt_emb = self.dropout(self.embedding(tgt))

        # Generate masks if not provided
        if src_mask is None:
            # Generate mask based on src length of tgt for alignment
            src_mask = self.generate_mask(src[:, :tgt.shape[1]])
        if tgt_mask is None:
            tgt_mask = self.generate_decoder_mask(tgt)

        # Removing the sequence length adjustment
        #src_mask = src_mask[:, :, :tgt.shape[1]] # Adjust src_mask's sequence length

        #src_mask = src_mask.unsqueeze(1)  # Add a dimension for heads (if necessary)
        #tgt_mask = tgt_mask.unsqueeze(1)  # Add a dimension for heads (if necessary)

        # Align src_emb with target sequence length before passing to encoder
        src_emb = src_emb[:, :tgt.shape[1], :]

        encoder_out = self.encoder(src_emb, src_mask)
        decoder_out = self.decoder(tgt_emb, encoder_out, src_mask, tgt_mask)

        return self.fc_out(decoder_out)

    def generate_mask(self, sequence):
      # Get the padding token ID
      pad_id = sp_en.piece_to_id('<pad>')
      # Create a mask where padding tokens are 0, others are 1
      mask = (sequence != pad_id)  # shape: (batch_size, sequence_length)
      # Add dimensions for broadcasting (unsqueeze once)
      mask = mask.unsqueeze(1)  # shape: (batch_size, 1, sequence_length)

      return mask.type(torch.float32)  # or torch.float32, depending on your requirements

    def generate_decoder_mask(self, tgt):
        # Get the padding token ID
        pad_id = sp_en.piece_to_id('<pad>')
        # Create a mask where padding tokens are 0, others are 1
        mask = (tgt != pad_id)  # shape: (batch_size, sequence_length)
        # Add dimensions for broadcasting (unsqueeze once)
        mask = mask.unsqueeze(1)  # shape: (batch_size, 1, sequence_length)

        # Changed the mask type to float32 instead of bool
        mask = mask.type(torch.float32)
        # Create a subsequent mask (triangular mask)
        seq_len = tgt.size(1)  # Get the target sequence length
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.float32) # shape: (sequence_length, sequence_length)
        # Expand dimensions of subsequent_mask to match the padding mask
        subsequent_mask = subsequent_mask.unsqueeze(0)
        # Combine padding mask and subsequent mask
        mask = mask * subsequent_mask

        return mask


###############
#### TRAIN ####
###############
# Supposons que tu as déjà défini le modèle Transformer (comme montré précédemment)
model = Transformer(vocab_size=len(sp_en), embed_dim=256, num_layers=6, num_heads=8, ff_dim=512, dropout=dropout_rate)
print(f"vocab_size = {vocab_size}")
# Définir un optimiseur (par exemple Adam)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Définir une fonction de perte (par exemple CrossEntropy pour la traduction)
criterion = nn.CrossEntropyLoss(ignore_index=sp_en.pad_id())  # Ignorer le PAD token pendant le calcul de la perte

# Mettre le modèle en mode entraînement
model.train()

# Boucle d'entraînement
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0  # Variable pour suivre la perte totale sur un epoch

    for i, (src, tgt) in enumerate(dataloader):
        optimizer.zero_grad()  # Remettre à zéro les gradients
        #print(f"src shape: {src.shape}")   # Should be (batch_size, seq_len)
        #print(f"tgt shape: {tgt.shape}")   # Should be (batch_size, seq_len)
        #print(f"tgt[:, :-1] shape: {tgt[:, :-1].shape}")  # Should be (batch_size, seq_len - 1)
        #print(f"tgt[:, 1:] shape: {tgt[:, 1:].shape}")    # Should also be (batch_size, seq_len - 1)
        # Passer les entrées à travers le modèle
        output = model(src, tgt[:, :-1])  # Entrée : src, sortie : tgt décalé d'une position (pour prédire le mot suivant)

        # Calculer la perte
        # Utilisation de la dernière colonne de la sortie (cible) pour le calcul de la perte
        loss = criterion(output.view(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))  # La sortie sans le token de début, et le target sans le token de début
        
        # Calculer les gradients et mettre à jour les poids
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:  # Afficher la perte tous les 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {total_loss/100:.4f}")
            total_loss = 0  # Réinitialiser la perte
            #print(f"Last output shape = {output.shape}")
            #print(f"Last output = {output}")

    # Affichage de la perte à la fin de chaque époque
    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

####################
#### Evaluation ####
####################

model.eval()
# 1. Preprocess the input sentence
input_sentence = "Hello, how are you?"  # Simple English sentence to translate
input_tokens = tokenize(input_sentence, sp_en)  # Tokenize using the English SentencePiece processor
input_indices = tokens_to_indices(input_tokens, sp_en)  # Convert to indices
input_indices = pad_sequence(input_indices, max_len, sp_en)  # Pad the sequence to max_len

# 2. Feed the input sentence to the model
input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Add batch dimension (shape: [1, max_len])

# Create a decoder input (usually just the <sos> token for start of sentence)
sos_token = sp_fr.piece_to_id('<sos>')
decoder_input = torch.tensor([sos_token] * (max_len-1)).unsqueeze(0)  # Add batch dimension and pad decoder input to max_len
print(decoder_input.shape)

# Perform the forward pass (assuming model.forward(src, tgt) works with the model)
with torch.no_grad():  # Disable gradient computation for evaluation
    print(f"input tensor = {input_tensor}")
    print(f"decoder input = {decoder_input}")
    output = model(input_tensor, decoder_input)  # Pass the input tensor through the encoder-decoder model
print(f"output shape = {output.shape}")
print(f"output print = {output}")
# 3. Decode the output tokens
output_indices = output.argmax(dim=-1).squeeze().tolist()  # Get the token indices with the highest probability
output_tokens = [sp_fr.id_to_piece(idx) for idx in output_indices if idx != sp_fr.piece_to_id('<pad>')]  # Decode tokens (remove <pad>)
print(f"output tokens = {output_tokens}")

second_column = output[0, 1, :]
# Print the second column
print(second_column)
print(output)
# 4. Post-process the output
translated_sentence = " ".join(output_tokens)  # Join tokens to form the translated sentence
print("Translated sentence:", translated_sentence)
