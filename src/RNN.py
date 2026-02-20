import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, epochs=10, device='cpu', early_stopping_patience=10):
    """Train the model with proper tracking and device support"""
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss_train = 0
        total_mae_train = 0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs).squeeze(1)
            
            loss = criterion(predictions, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss_train += loss.item()
            total_mae_train += torch.abs(predictions - labels).mean().item()

        # Validation phase
        model.eval()
        total_loss_val = 0
        total_mae_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs).squeeze(1)
                
                loss = criterion(predictions, labels)
                total_loss_val += loss.item()
                total_mae_val += torch.abs(predictions - labels).mean().item()

        # Calculate averages
        avg_train_loss = total_loss_train / len(train_dataloader)
        avg_val_loss = total_loss_val / len(val_dataloader)
        avg_train_mae = total_mae_train / len(train_dataloader)
        avg_val_mae = total_mae_val / len(val_dataloader)

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + total_loss_train / len(train_dataloader))
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['learning_rates'].append(current_lr)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | '
                  f'LR: {current_lr:.6f}')
        
        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nLoaded best model with validation loss: {best_val_loss:.4f}')

    return history


def evaluate_model(model, test_dataloader, criterion, device='cpu', tolerance=0.3):
    """Evaluate model on test set"""
    model = model.to(device)
    model.eval()
    
    total_loss = 0
    total_mae = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs).squeeze(1)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            total_mae += torch.abs(predictions - labels).mean().item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    
    # Calculate R² score
    ss_res = np.sum((all_labels - all_predictions) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    # Regression accuracy with tolerance
    test_accuracy = np.mean(np.abs(all_predictions - all_labels) < tolerance)
    
    # RMSE
    rmse = np.sqrt(np.mean((all_predictions - all_labels) ** 2))
    
    # MAPE (Mean Absolute Percentage Error) - avoid division by zero
    mape = np.mean(np.abs((all_labels - all_predictions) / (np.abs(all_labels) + 1e-8))) * 100
    
    print(f'\nTest Metrics:')
    print(f'  Loss (MSE): {avg_loss:.4f}')
    print(f'  MAE: {avg_mae:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R² Score: {r2_score:.6f}')
    print(f'  MAPE: {mape:.2f}%')
    print(f'  Accuracy (tolerance={tolerance}): {test_accuracy*100:.2f}%')
    
    return {
        'test_loss': avg_loss,
        'test_mae': avg_mae,
        'test_rmse': rmse,
        'r2_score': r2_score,
        'mape': mape,
        'test_accuracy': test_accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[0, 1].plot(history['train_mae'], label='Train MAE', marker='o', markersize=3)
    axes[0, 1].plot(history['val_mae'], label='Val MAE', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Training and Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'learning_rates' in history and len(history['learning_rates']) > 0:
        axes[1, 0].plot(history['learning_rates'], marker='o', markersize=3, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss comparison
    axes[1, 1].plot(np.array(history['train_loss']) - np.array(history['val_loss']), 
                    marker='o', markersize=3, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Loss - Val Loss')
    axes[1, 1].set_title('Overfitting Monitor')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_predictions(test_results, num_samples=100):
    """Plot predictions vs actual values"""
    predictions = test_results['predictions'][:num_samples]
    labels = test_results['labels'][:num_samples]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    ax1.scatter(labels, predictions, alpha=0.6, s=30)
    ax1.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs Actual Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = predictions - labels
    ax2.scatter(labels, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def main():
    # Example usage with improved training
    print("="*60)
    print("Linear Regression Model Training")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 5000  # Increased dataset size
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create a more complex relationship
    Y = (2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 
         0.5 * X[:, 3] * X[:, 4] + np.random.randn(n_samples) * 0.1)
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Target range: [{Y.min():.2f}, {Y.max():.2f}]")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create data module and get dataloaders with normalization
    data_module = LinearRegressionDataModule(X, Y, normalize=True)
    train_loader, val_loader, test_loader = data_module.get_dataloaders(
        batch_size=128, test_size=0.15, val_size=0.15
    )
    
    print(f"\nData splits:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model with improved architecture
    model = LinearRegressionModel(
        input_size=n_features, 
        hidden_neurons=512,
        dropout_rate=0.15
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with improved settings
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.002, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Learning rate scheduler with cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Train model
    print("\n" + "="*60)
    print("Training started...")
    print("="*60)
    
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler=scheduler, epochs=300, device=device, 
        early_stopping_patience=30
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_results = evaluate_model(model, test_loader, criterion, device=device, tolerance=0.3)
    
    # Plot predictions
    print("\nPlotting predictions...")
    plot_predictions(test_results, num_samples=200)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'test_results': {k: v for k, v in test_results.items() if k not in ['predictions', 'labels']}
    }, 'linear_regression_model.pth')
    print("\nModel saved to 'linear_regression_model.pth'")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

def load_model(model_path, input_size, hidden_neurons=512, dropout_rate=0.15, device='cpu'):
    """Load model and optimizer state from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        input_size: Number of input features (required to reconstruct model)
        hidden_neurons: Number of hidden neurons (default: 512)
        dropout_rate: Dropout rate (default: 0.15)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded model
        history: Training history (if available)
        test_results: Test results (if available)
    """
    # Load checkpoint with weights_only=False for backward compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model instance with correct architecture
    model = LinearRegressionModel(
        input_size=input_size,
        hidden_neurons=hidden_neurons,
        dropout_rate=dropout_rate
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load additional information if available
    history = checkpoint.get('history', None)
    test_results = checkpoint.get('test_results', None)
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Device: {device}")
    if test_results:
        print(f"Previous test results:")
        for key, value in test_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    return model, history, test_results


def predict(model, X, device='cpu', data_module=None):
    model = model.to(device)
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    
    X_tensor = X_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
    
    # Denormalize if data module is provided
    if data_module is not None and hasattr(data_module, 'denormalize_predictions'):
        predictions = data_module.denormalize_predictions(predictions)
    
    return predictions