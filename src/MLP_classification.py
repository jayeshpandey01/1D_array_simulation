import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split

# use leakly relu activation function for MLP classifier on 1D signal data using pytorch
class MLPClassifierTorch:
    def __init__(self, input_size, num_classes, hidden_layers=(128, 64), activation=torch.nn.LeakyReLU(negative_slope=0.1), lr=1e-3, device=None):
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        in_features = input_size
        for h in hidden_layers:
            layers.append(torch.nn.Linear(in_features, h))
            layers.append(torch.nn.ReLU())
            in_features = h
        layers.append(torch.nn.Linear(in_features, num_classes))
        self.net = torch.nn.Sequential(*layers).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def fit(self, X_train, y_train, epochs=20, batch_size=32):
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.long).to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                self.opt.zero_grad()
                logits = self.net(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.opt.step()

    def predict(self, X_test):
        self.net.eval()
        X = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.net(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return (preds == y_test).mean()
    
# signal dataset class for pytorch
class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def main():
    df = pd.read_csv('../datasets/train_waveforms.csv')
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPClassifierTorch(input_size=X.shape[1], num_classes=len(np.unique(y)))
    model.fit(X_train, y_train)
    print("Test Accuracy:", model.score(X_test, y_test))