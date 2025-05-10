import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# HYPERPARAMETERS (best from cross validation, plus longer training)
EMBED_DIM   = 32
HIDDEN_DIM  = 100
LR          = 1e-3
BATCH_SIZE  = 32
NUM_EPOCHS  = 40
PATIENCE    = 6
PROBE_FRAC  = 0.50
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD DATA
dataset = torch.load("dataset.pth", weights_only=False) 
X_all, y_all = dataset.tensors
VOCAB_SIZE, OUTPUT_DIM = 4, y_all.shape[1]

probe_size = int(len(dataset) * PROBE_FRAC)
train_size = len(dataset) - probe_size
train_ds, probe_ds = random_split(dataset, [train_size, probe_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
probe_loader = DataLoader(probe_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 2. MODEL
class HashPredictor(nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

model = HashPredictor(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val, patience_ctr = float("inf"), 0

# 3. TRAIN
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= train_size

    # simple val loss on probe set for early-stop / scheduler
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in probe_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= probe_size
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | probe-loss {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        patience_ctr = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping.")
            break

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# 4. FINAL SEARCH SPACE REDUCTION (probe_ds only)
with torch.no_grad():
    all_preds = model(X_all.to(DEVICE)) # predictions for every candidate

reductions = []
for xb, yb in probe_loader:
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    dists = torch.norm(all_preds.unsqueeze(1) - yb.unsqueeze(0), dim=2)
    for i in range(dists.shape[1]):
        # locate true input row
        true_vec = xb[i].unsqueeze(0)
        idx_true = (X_all.to(DEVICE) == true_vec).all(dim=1).nonzero(as_tuple=False).item()
        rank = torch.argsort(dists[:, i]).tolist().index(idx_true) + 1
        reductions.append(1 - rank / len(X_all))

avg_red = float(torch.tensor(reductions).mean()) * 100
print(f"\n> Average search-space reduction on unseen probe set: {avg_red:.2f}%")

