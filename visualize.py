# --- visualize.py ---
import matplotlib.pyplot as plt
import numpy as np
from train import model, text_to_tensor, MAX_LEN, vocab
from nltk.tokenize import word_tokenize
import torch

model.eval()

def visualize_attention(text, top_k=1):
    tokens = word_tokenize(text.lower())[:MAX_LEN]
    seq_len = len(tokens)
    x = text_to_tensor(text, vocab).unsqueeze(0).to(model.device)
    _, attn_maps = model(x)

    for layer_idx, attn in enumerate(attn_maps):
        # attn shape: [batch, seq_len, seq_len]
        attn = attn[0].detach().cpu().numpy()  # [seq_len, seq_len]
        trimmed_attn = attn[:seq_len, :seq_len]

        # Print top-k token relations
        print(f"\nLayer {layer_idx+1} Attention:")
        for i in range(seq_len):
            top_keys = trimmed_attn[i].argsort()[-top_k:][::-1]
            for j in top_keys:
                print(f"'{tokens[i]}' attends to '{tokens[j]}' (weight: {trimmed_attn[i][j]:.2f})")

        # Plot heatmap
        # TODO: FIX MAPPING LOGIC
        plt.figure(figsize=(6, 5))
        plt.title(f"Layer {layer_idx+1} Attention")
        plt.imshow(trimmed_attn, cmap='viridis', aspect='auto')
        plt.xticks(ticks=np.arange(seq_len), labels=tokens, rotation=90)
        plt.yticks(ticks=np.arange(seq_len), labels=tokens)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

# Example usage:
# visualize_attention("this movie is really good")
