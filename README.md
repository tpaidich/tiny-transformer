# Tiny Transformer

A minimal Transformer encoder built from scratch in PyTorch — designed to demystify how attention works. This project includes training on real sentiment data and visualizing learned attention patterns across layers.

- Custom Self-Attention Layer
- Multi-layer Encoder with LayerNorm and Feedforward blocks
- Word-level tokenization using NLTK
- Attention heatmap visualizations that show what the model "focuses" on

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Download imdb.csv and place it inside the data/ directory.

### 3. Train the model
```bash
python train.py
```

### 4. Visualize attention
```bash
python -i visualize.py
>>> visualize_attention("this movie is awesome!")
```

---

## Example output

Running visualize_attention("this movie is awesome!") will:

- Print top tokens each word attends to
- Generate a heatmap showing token-to-token attention
- This helps explain how transformers interpret language.


  
