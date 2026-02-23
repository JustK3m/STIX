import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ✅ Dummy dataset (à remplacer par tes vraies simulations)
def generate_fake_data(num_samples, counts_size, srm_size, photon_size):
    X = []
    y = []
    for _ in range(num_samples):
        counts = np.random.uniform(10, 100, size=(counts_size,))
        srm = np.random.uniform(0, 1, size=(counts_size, photon_size))
        photons = np.random.uniform(1, 10, size=(photon_size,))
        # input = counts + SRM aplati
        x_input = np.concatenate([counts, srm.flatten()])
        X.append(x_input)
        y.append(photons)
    return np.array(X), np.array(y)

# ✅ Hyperparamètres
NUM_SAMPLES = 10000
COUNTS_SIZE = 30
PHOTON_SIZE = 1028
SRM_SIZE = (COUNTS_SIZE, PHOTON_SIZE)

# ✅ Données
X, y = generate_fake_data(NUM_SAMPLES, COUNTS_SIZE, SRM_SIZE, PHOTON_SIZE)

# ✅ Dataset PyTorch
class STIXDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = STIXDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ Réseau simple
class STIXNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.net(x)

model = STIXNN(X.shape[1], y.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ✅ Entraînement
for epoch in range(100):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item()}")

# ✅ Sauvegarde
torch.save(model, "model.pt")
print("✅ Modèle sauvegardé dans model.pt")


