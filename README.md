# tiny-transformer-demo
A tiny Transformer encoder built from scratch using PyTorch. 

This model performs simple text classification (positive vs. negative sentiment) using an IMDB dataset for character sentiment analysis of word-level inputs. It includes attention visualizations.

---
To install and run:
```bash
pip install -r requirements.txt
python train.py
```

To visualize attention:
```bash
python -i visualize.py
>>> visualize_attention("this movie is great")
```
---

