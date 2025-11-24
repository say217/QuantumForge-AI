# trainer.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from rich.progress import track
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
  
    def train(self, model, train_loader, test_loader):
        if self.config.loss == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        scaler = GradScaler() if torch.cuda.is_available() else None
        best_loss = float('inf')
        train_losses, test_losses = [], []
        patience, early_stopping = 15, False
        patience_counter = 0
        start_time = time.time()
        os.makedirs("models", exist_ok=True)
      
        for epoch in track(range(self.config.epochs), description="Training epochs..."):
            if early_stopping:
                break
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                if scaler:
                    with autocast():
                        output = model(X)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                train_loss += loss.item() * X.size(0)
            train_loss /= len(train_loader.dataset)
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if scaler:
                        with autocast():
                            output = model(X)
                            batch_loss = criterion(output, y).item()
                    else:
                        output = model(X)
                        batch_loss = criterion(output, y).item()
                    test_loss += batch_loss * X.size(0)
            test_loss /= len(test_loader.dataset)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            scheduler.step(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), f'models/{self.config.ticker}_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    early_stopping = True
              
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Time: {time.time() - epoch_start:.2f}s")
      
        print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
        return train_losses, test_losses
  
    def evaluate(self, model, loader, data_handler):
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                output = model(X).cpu().numpy()
                preds.extend(data_handler.inverse_target_transform(output))
                actuals.extend(data_handler.inverse_target_transform(y.cpu().numpy()))
        return np.array(preds), np.array(actuals)
  
    def directional_accuracy(self, actuals, preds):
        if len(actuals) <= 1 or len(preds) <= 1:
            return 0.0
        actual_diff = np.diff(actuals)
        pred_diff = np.diff(preds)
        correct = np.sum((actual_diff > 0) == (pred_diff > 0))
        return correct / len(actual_diff) * 100.0