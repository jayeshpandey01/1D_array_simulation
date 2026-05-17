"""
Model comparison for signal parameter regression.

Expected CSV schema (from precise_dataset or dataset.save_to_csv):
  - Metadata: wave_type, frequency, amplitude, phase, snr_db (optional)
  - Signals: signal_0 .. signal_{N-1}  (default N=256)
  - target_col for regression: 'frequency' | 'amplitude' | 'phase'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.constants import SAMPLE_RATE
from src.features import extract_batch_features
from src.identify_amplitude_frequency import analyze_signal_full
from src.signal_io import signal_column_names

class SignalModelComparator:
    def __init__(self, df, target_col='amplitude', random_state=42):
        """
        Initialize the comparator with data and target.
        Args:
            df: DataFrame containing signal_0 to signal_255 and target columns.
            target_col: The column name to predict (e.g., 'amplitude' or 'frequency').
            random_state: Seed for reproducibility.
        """
        self.df = df
        self.target_col = target_col
        self.random_state = random_state
        self.signal_cols = signal_column_names(self.df)
        if not self.signal_cols:
            raise ValueError("DataFrame has no signal_* columns.")

        self.X_raw = self.df[self.signal_cols].values
        self.y = self.df[self.target_col].values
        
        # Train-Test Split (Raw)
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            self.X_raw, self.y, test_size=0.2, random_state=self.random_state
        )
        
        self.results = {}
        self.cnn_history = None
        self.best_model_name = None
        self.best_model = None

    def extract_features(self, data):
        """Delegate to src.features (single FFT implementation)."""
        return extract_batch_features(data, sampling_rate=SAMPLE_RATE)

    def train_classical_models(self):
        """Train and evaluate scikit-learn models using 5-fold CV."""
        print(f"--- Training Classical Models for {self.target_col} ---")
        
        # Feature Extraction
        X_train_feats = self.extract_features(self.X_train_raw)
        X_test_feats = self.extract_features(self.X_test_raw)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feats)
        X_test_scaled = scaler.transform(X_test_feats)
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=self.random_state),
            'SVR': SVR(kernel='rbf'),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
        }
        
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            cv_results = cross_validate(model, X_train_scaled, self.y_train, cv=5, scoring=scoring)
            
            # Fit final model on all training data for test eval
            model.fit(X_train_scaled, self.y_train)
            test_preds = model.predict(X_test_scaled)
            
            self.results[name] = {
                'Test R2': r2_score(self.y_test, test_preds),
                'Test MAE': mean_absolute_error(self.y_test, test_preds),
                'Test RMSE': np.sqrt(mean_squared_error(self.y_test, test_preds)),
                'model_obj': model,
                'scaler': scaler
            }

    def _get_cnn_model(self):
        """Define 1D CNN Model (requires PyTorch)."""
        import torch
        import torch.nn as nn

        class CNN1D(nn.Module):
            def __init__(self):
                super(CNN1D, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                )
                self.fc_layers = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                x = self.fc_layers(x)
                return x
        return CNN1D()

    def train_pytorch_cnn(self, epochs=50, batch_size=32):
        """Train 1D CNN using PyTorch."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        print(f"--- Training 1D CNN for {self.target_col} ---")

        # Scaling raw signals for CNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train_raw)
        X_test_scaled = scaler.transform(self.X_test_raw)
        
        # Convert to Tensors
        X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
        y_tr_t = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
        y_te_t = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tr_t, y_tr_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = self._get_cnn_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        history = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            history.append(avg_loss)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.cnn_history = history
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            preds_t = model(X_te_t)
            preds = preds_t.numpy().flatten()
            
        self.results['PyTorch_CNN'] = {
            'Test R2': r2_score(self.y_test, preds),
            'Test MAE': mean_absolute_error(self.y_test, preds),
            'Test RMSE': np.sqrt(mean_squared_error(self.y_test, preds)),
            'model_obj': model,
            'scaler': scaler
        }

    def print_summary(self):
        """Print ranked comparison of model performance."""
        summary = []
        for name, metrics in self.results.items():
            summary.append({
                'Model': name,
                'Test R2': metrics['Test R2'],
                'Test MAE': metrics['Test MAE'],
                'Test RMSE': metrics['Test RMSE']
            })
        
        res_df = pd.DataFrame(summary).sort_values(by='Test R2', ascending=False)
        print("\n--- Model Comparison Summary ---")
        print(res_df.to_string(index=False))
        
        self.best_model_name = res_df.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]
        return res_df

    def plot_best_model(self, num_samples=5):
        """Plot original signal vs prediction for the best model."""
        if not self.best_model_name:
            print("No models trained yet.")
            return
            
        print(f"Plotting Predictions for Best Model: {self.best_model_name}")
        
        # Prepare inputs
        X_test_sample = self.X_test_raw[:num_samples]
        y_test_sample = self.y_test[:num_samples]
        
        if self.best_model_name == 'PyTorch_CNN':
            import torch

            X_scaled = self.best_model['scaler'].transform(X_test_sample)
            X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
            self.best_model['model_obj'].eval()
            with torch.no_grad():
                preds = self.best_model['model_obj'](X_t).numpy().flatten()
        else:
            X_feats = self.extract_features(X_test_sample)
            X_scaled = self.best_model['scaler'].transform(X_feats)
            preds = self.best_model['model_obj'].predict(X_scaled)
            
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_sample, 'bo-', label='Actual')
        plt.plot(preds, 'rx--', label='Predicted')
        plt.title(f"{self.target_col} Prediction using {self.best_model_name}")
        plt.xlabel("Sample Index")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cnn_loss(self):
        """Show training loss curve for the CNN."""
        if self.cnn_history is None:
            print("CNN has not been trained yet.")
            return
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.cnn_history, label='Training Loss (MSE)')
        plt.title('PyTorch 1D CNN Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_fft_baseline(self) -> dict:
        """
        Pipeline analyzer baseline (same path as GUI/evaluate).
        Uses analyze_signal_full per test sample.
        """
        preds = []
        for signal in self.X_test_raw:
            res = analyze_signal_full(signal, sampling_rate=SAMPLE_RATE)
            if self.target_col == "frequency":
                preds.append(res.frequency_hz if res.frequency_hz is not None else np.nan)
            elif self.target_col == "amplitude":
                preds.append(res.amplitude if res.amplitude is not None else np.nan)
            else:
                preds.append(res.phase_rad if res.phase_rad is not None else np.nan)

        preds = np.array(preds, dtype=np.float64)
        valid = np.isfinite(preds) & np.isfinite(self.y_test)
        if valid.sum() == 0:
            metrics = {"Test R2": np.nan, "Test MAE": np.nan, "Test RMSE": np.nan}
        else:
            metrics = {
                "Test R2": r2_score(self.y_test[valid], preds[valid]),
                "Test MAE": mean_absolute_error(self.y_test[valid], preds[valid]),
                "Test RMSE": np.sqrt(mean_squared_error(self.y_test[valid], preds[valid])),
            }
        self.results["FFT_Pipeline"] = {**metrics, "predictions": preds}
        return metrics

    @staticmethod
    def plot_prediction_errors(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str = "target",
        save_path=None,
    ) -> None:
        """Scatter detected vs. actual and error histogram."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=15)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, "r--")
        axes[0].set_xlabel(f"Actual {target_name}")
        axes[0].set_ylabel(f"Predicted {target_name}")
        axes[0].set_title(f"{target_name}: predicted vs. actual")
        axes[0].grid(True, alpha=0.3)

        err = y_pred - y_true
        axes[1].hist(err, bins=40, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="r", linestyle="--")
        axes[1].set_xlabel("Prediction error")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"{target_name}: error distribution")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

# Example Usage Template:
# comparator = SignalModelComparator(df, target_col='amplitude')
# comparator.train_classical_models()
# comparator.train_pytorch_cnn(epochs=50)
# comparator.print_summary()
# comparator.plot_cnn_loss()
# comparator.plot_best_model()
