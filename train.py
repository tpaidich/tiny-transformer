import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformer.transformer_encoder import TinyTransformer

VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
VOCAB_SIZE = len(VOCAB)
EMBED_DIM = 32
FF_DIM = 64
NUM_LAYERS = 2
MAX_LEN = 50
BATCH_SIZE = 8
EPOCHS = 20
NUM_CLASSES = 2

examples = [
    ("i love this movie", 1),
    ("this film is great", 1),
    ("what a fantastic story", 1),
    ("absolutely wonderful", 1),
    ("i hate this", 0),
    ("this movie is bad", 0),
    ("what a terrible film", 0),
    ("horrible experience", 0)
] * 20
random.shuffle(examples)

char2idx = {ch: i for i, ch in enumerate(VOCAB)}

def text_to_tensor(text):
    idxs = [char2idx.get(c, 0) for c in text.lower()[:MAX_LEN]]
    if len(idxs) < MAX_LEN:
        idxs += [0] * (MAX_LEN - len(idxs))
    return torch.tensor(idxs)

class ToyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return text_to_tensor(x), torch.tensor(y)

train_loader = DataLoader(ToyDataset(examples), batch_size=BATCH_SIZE, shuffle=True)

model = TinyTransformer(VOCAB_SIZE, EMBED_DIM, FF_DIM, NUM_LAYERS, NUM_CLASSES, MAX_LEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y_batch).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Loss = {total_loss:.3f}, Acc = {acc:.2f}")
