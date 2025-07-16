# --- transformer/transformer_encoder.py ---
import torch
import torch.nn as nn
from transformer.encoder_layer import EncoderLayer

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, max_len, num_classes, ff_dim=128, dropout=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, ff_dim)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)

        x_cls = x[:, 0]  # use the first token (CLS-like)
        return self.classifier(x_cls), attn_maps
