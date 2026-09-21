import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 8
PRED_LEN = 2
BATCH_SIZE = 32
EPOCHS = 1500
LEARNING_RATE = 0.001
L2_WEIGHT_DECAY = 1e-4

csv_path = '../data_preprocess/best_track_records_p6.csv'
sp_json_path = '../data_preprocess/sp_data_matrix.json'
sst_json_path = '../data_preprocess/sst_data_matrix.json'

data = pd.read_csv(csv_path)
with open(sp_json_path, 'r') as f:
    sp_data = json.load(f)
with open(sst_json_path, 'r') as f:
    sst_data = json.load(f)

grouped = data.groupby('Storm Name')
latitude_scaler = MinMaxScaler()
longitude_scaler = MinMaxScaler()


def compute_diff(group):
    group['Latitude_Diff'] = group['Latitude (°N)'].diff().fillna(0)
    group['Longitude_Diff'] = group['Longitude (°E)'].diff().fillna(0)
    return group

data = grouped.apply(compute_diff)

def normalize_grid(grid):
    grid_min = np.min(grid)
    grid_max = np.max(grid)
    return (grid - grid_min) / (grid_max - grid_min)

def create_samples(group):
    samples = []
    for i in range(len(group) - 9):
        seq = group.iloc[i:i+10]

        inputs = seq.iloc[:8][['Latitude_Diff', 'Longitude_Diff']].values

        targets = seq.iloc[8:][['Latitude_Diff', 'Longitude_Diff']].values
        datetime_keys = seq['DateTime(UTC)'][:8].tolist()

        sp_grids = [sp_data[f"{seq['Storm Name'].iloc[0]}{dt}"]['sp_grid'] for dt in datetime_keys]
        sst_grids = [sst_data[f"{seq['Storm Name'].iloc[0]}{dt}"]['sst_grid'] for dt in datetime_keys]
        

        sp_grids = np.array([normalize_grid(sp) for sp in sp_grids])
        sst_grids = np.array([normalize_grid(sst) for sst in sst_grids])
        

        sp_grids_flat = sp_grids.reshape(sp_grids.shape[0], -1) 
        sst_grids_flat = sst_grids.reshape(sst_grids.shape[0], -1) 
        

        inputs = np.concatenate([inputs, sp_grids_flat, sst_grids_flat], axis=1) 
        
        samples.append((inputs, targets))
    return samples

class EarlyStopping:
    def __init__(self, patience=30, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


data = data.reset_index(drop=True)
data['Latitude_Diff'] = latitude_scaler.fit_transform(data[['Latitude_Diff']])
data['Longitude_Diff'] = longitude_scaler.fit_transform(data[['Longitude_Diff']])
samples = []
for _, group in data.groupby('Storm Name'):
    samples.extend(create_samples(group))

class TyphoonDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, targets = self.samples[idx]
        return (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

torch.manual_seed(2025)

# # Split data into train and test
train_samples, test_samples = train_test_split(samples, test_size=0.1, random_state=42)
train_dataset = TyphoonDataset(train_samples)
test_dataset = TyphoonDataset(test_samples)
input_dim = train_samples[0][0].shape[1]
hidden_dim = 256
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define the CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, cnn_channels=256, kernel_size=3, lstm_layers=3):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(input_size=cnn_channels * 4, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, PRED_LEN),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.cnn(x)
        x = x.transpose(1, 2)  
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] 
        output = self.fc(last_out)
        return output



def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets[:, :, 0].squeeze(-1)  # For latitude
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def train_model(model, train_loader, criterion, optimizer, is_latitude=True):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets[:, :, 0].squeeze(-1) if is_latitude else targets[:, :, 1].squeeze(-1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test_model(model, test_loader, criterion, scaler):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            

            targets = targets[:, :, 0].squeeze(-1)
            

            targets_raw = scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1))
            targets_raw = targets_raw.reshape(targets.shape) 
            
            outputs = model(inputs)
            outputs_raw = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
            outputs_raw = outputs_raw.reshape(outputs.shape) 
            

            loss = criterion(
                torch.tensor(outputs_raw, dtype=torch.float32).to(device),
                torch.tensor(targets_raw, dtype=torch.float32).to(device)
            )
            total_loss += loss.item()
    return total_loss / len(test_loader)




if __name__ == "__main__":

    lat_model = CNNLSTM(input_dim, hidden_dim).to(device)
    lon_model = CNNLSTM(input_dim, hidden_dim).to(device)

    lat_criterion = nn.MSELoss()
    lon_criterion = nn.MSELoss()
    lat_optimizer = torch.optim.Adam(lat_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)
    lon_optimizer = torch.optim.Adam(lon_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)

    lat_early_stopping = EarlyStopping(patience=30)
    lon_early_stopping = EarlyStopping(patience=30)

    count = 0
    for epoch in range(1, EPOCHS + 1):

        train_lat_loss = train_model(lat_model, train_loader, lat_criterion, lat_optimizer)
        test_lat_loss = test_model(lat_model, test_loader, lat_criterion, latitude_scaler)
        

        train_lon_loss = train_model(lon_model, train_loader, lon_criterion, lon_optimizer)
        test_lon_loss = test_model(lon_model, test_loader, lon_criterion, longitude_scaler)

        count += 1
        if count % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Train Latitude Loss: {train_lat_loss:.4f}, Test Latitude Loss: {test_lat_loss:.4f}")
            print(f"Epoch {epoch}/{EPOCHS}, Train Longitude Loss: {train_lon_loss:.4f}, Test Longitude Loss: {test_lon_loss:.4f}")

        if count>=510:
            lat_early_stopping(test_lat_loss)
            lon_early_stopping(test_lon_loss)

        if lat_early_stopping.early_stop and lon_early_stopping.early_stop:
            print("Early stopping triggered for both models.")
            break


    torch.save(lat_model, "model/lat_cnn_lstm_model.pth")
    torch.save(lon_model, "model/lon_cnn_lstm_model.pth")
    


