# --- train.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nltk.tokenize import word_tokenize
import nltk
from transformer.transformer_encoder import TinyTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download('punkt_tab')

# Hyperparameters
MAX_LEN = 50
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 5

# Load Dataset
def load_imdb_data(csv_path="data/imdb.csv"):
    df = pd.read_csv(csv_path)
    df = df[["review", "sentiment"]]
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_data = list(zip(train_df["review"], train_df["label"]))
    test_data = list(zip(test_df["review"], test_df["label"]))
    return train_data, test_data

# Build Vocabulary
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for sentence, _ in dataset:
        tokens = word_tokenize(sentence.lower())
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

# Convert text to tensor
def text_to_tensor(text, vocab, max_len=MAX_LEN):
    tokens = word_tokenize(text.lower())[:max_len]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids)

# Prepare dataset tensors
def prepare_data(data, vocab):
    inputs = []
    labels = []
    for text, label in data:
        inputs.append(text_to_tensor(text, vocab))
        labels.append(torch.tensor(label))
    return TensorDataset(torch.stack(inputs), torch.tensor(labels))

# Load and prepare data
train_data, test_data = load_imdb_data()
vocab = build_vocab(train_data + test_data)
train_dataset = prepare_data(train_data, vocab)
test_dataset = prepare_data(test_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model
model = TinyTransformer(vocab_size=len(vocab), embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, max_len=MAX_LEN, num_classes=NUM_CLASSES)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    acc = total_correct / total
    print(f"Epoch {epoch}: Loss = {avg_loss:.3f}, Acc = {acc:.2f}")

# Optional: Save model
# torch.save(model.state_dict(), 'tiny_transformer_sst2.pt')