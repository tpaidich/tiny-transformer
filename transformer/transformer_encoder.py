import torch.nn as nn
from transformer.embeddings import TokenEmbedding, PositionalEncoding
from transformer.encoder_layer import EncoderLayer

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, num_layers, num_classes, max_len):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, ff_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token), attn_maps
