from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib   # for loading scaler

app = FastAPI()

# Define model
class HGAT_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HGAT_GRU, self).__init__()
        self.attn = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.attn(x))
        h = self.dropout(h)
        h = h.unsqueeze(1)
        h, _ = self.gru(h)
        h = h.squeeze(1)
        out = self.fc(h)
        return out

# Load trained model weights
model = HGAT_GRU(input_dim=10, hidden_dim=32, output_dim=1)
model.load_state_dict(torch.load("models/hgat_gru_model.pth"))
model.eval()

# Load the saved scaler
scaler_y = joblib.load("models/scaler_y.pkl")

@app.get("/")
def root():
    return {"message": "HGAT+GRU backend is running"}

# Define request schema
class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Features):
    x = torch.tensor([data.features], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x)
        pred = scaler_y.inverse_transform(pred_scaled.numpy())
    return {"prediction": float(pred[0][0])}