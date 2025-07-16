import matplotlib.pyplot as plt
import numpy as np
from train import model, text_to_tensor, MAX_LEN

def visualize_attention(text):
    model.eval()
    x = text_to_tensor(text).unsqueeze(0).to(next(model.parameters()).device)
    _, attn_maps = model(x)
    tokens = list(text.lower()[:MAX_LEN])

    for i, attn in enumerate(attn_maps):
        attn = attn[0].detach().cpu().numpy()  # shape: (seq_len, seq_len)
        plt.figure(figsize=(6, 5))
        plt.title(f"Layer {i+1} Attention")
        plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90)
        plt.yticks(ticks=np.arange(len(tokens)), labels=tokens)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
