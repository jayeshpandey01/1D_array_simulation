"""
Test the trained linear regression model with integer inputs
"""
import numpy as np
import torch
from src.linear_regression import load_model, LinearRegressionDataModule

def test_with_integer_data(model_path='linear_regression_model.pth', 
                           input_size=10,
                           hidden_neurons=512,
                           dropout_rate=0.15):
    """
    Test the trained model with integer inputs
    
    Args:
        model_path: Path to the saved model
        input_size: Number of input features (must match training)
        hidden_neurons: Hidden layer size (must match training)
        dropout_rate: Dropout rate (must match training)
    """
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load the trained model
    print(f"\nLoading model from {model_path}...")
    model, history, test_results = load_model(
        model_path, 
        input_size=input_size,
        hidden_neurons=hidden_neurons,
        dropout_rate=dropout_rate,
        device=device
    )
    
    # Create integer test data
    print("\n" + "="*60)
    print("Creating integer test data...")
    print("="*60)
    
    # Example 1: Single integer sample
    integer_sample = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int32)
    print(f"\nTest Sample 1 (integers): {integer_sample[0]}")
    
    # Convert integers to float32 for the model
    float_sample = integer_sample.astype(np.float32)
    
    # Make prediction (without normalization - raw prediction)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(float_sample, dtype=torch.float32).to(device)
        raw_prediction = model(input_tensor).squeeze().cpu().numpy()
    
    print(f"Raw Prediction: {raw_prediction}")
    
    # Example 2: Multiple integer samples
    print("\n" + "="*60)
    print("Testing with multiple integer samples...")
    print("="*60)
    
    integer_samples = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ], dtype=np.int32)
    
    float_samples = integer_samples.astype(np.float32)
    
    with torch.no_grad():
        input_tensor = torch.tensor(float_samples, dtype=torch.float32).to(device)
        predictions = model(input_tensor).squeeze().cpu().numpy()
    
    print("\nInteger Inputs -> Predictions:")
    print("-" * 60)
    for i, (int_input, pred) in enumerate(zip(integer_samples, predictions)):
        print(f"Sample {i+1}: {int_input} -> {pred:.4f}")
    
    # Example 3: Random integer samples
    print("\n" + "="*60)
    print("Testing with random integer samples...")
    print("="*60)
    
    np.random.seed(42)
    random_integers = np.random.randint(-100, 100, size=(10, input_size), dtype=np.int32)
    random_floats = random_integers.astype(np.float32)
    
    with torch.no_grad():
        input_tensor = torch.tensor(random_floats, dtype=torch.float32).to(device)
        random_predictions = model(input_tensor).squeeze().cpu().numpy()
    
    print("\nRandom Integer Samples (first 5):")
    print("-" * 60)
    for i in range(min(5, len(random_integers))):
        print(f"Input: {random_integers[i]}")
        print(f"Prediction: {random_predictions[i]:.4f}\n")
    
    # Summary statistics
    print("="*60)
    print("Prediction Statistics:")
    print("="*60)
    print(f"Mean prediction: {random_predictions.mean():.4f}")
    print(f"Std prediction: {random_predictions.std():.4f}")
    print(f"Min prediction: {random_predictions.min():.4f}")
    print(f"Max prediction: {random_predictions.max():.4f}")
    
    return model, predictions


def test_with_normalized_integers(model_path='linear_regression_model.pth',
                                  input_size=10,
                                  hidden_neurons=512,
                                  dropout_rate=0.15):
    """
    Test with integers using proper normalization
    Note: This requires knowing the normalization parameters from training
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60)
    print("TESTING WITH NORMALIZATION")
    print("="*60)
    print("\nNote: For accurate predictions with normalization,")
    print("you need the mean and std from the training data.")
    print("This example shows the process.\n")
    
    # Load model
    model, _, _ = load_model(
        model_path,
        input_size=input_size,
        hidden_neurons=hidden_neurons,
        dropout_rate=dropout_rate,
        device=device
    )
    
    # Create integer test data
    integer_data = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ], dtype=np.int32)
    
    # Convert to float
    float_data = integer_data.astype(np.float32)
    
    # If you have the training statistics, normalize here
    # Example (you would need actual values from training):
    # X_mean = np.array([...])  # from training
    # X_std = np.array([...])   # from training
    # normalized_data = (float_data - X_mean) / X_std
    
    # For now, just use raw data
    with torch.no_grad():
        input_tensor = torch.tensor(float_data, dtype=torch.float32).to(device)
        predictions = model(input_tensor).squeeze().cpu().numpy()
    
    print("Integer inputs (converted to float):")
    for i, (int_input, pred) in enumerate(zip(integer_data, predictions)):
        print(f"  {int_input} -> {pred:.4f}")
    
    return predictions


if __name__ == "__main__":
    print("="*60)
    print("LINEAR REGRESSION MODEL - INTEGER INPUT TESTING")
    print("="*60)
    
    # Test with integers
    try:
        model, predictions = test_with_integer_data(
            model_path='linear_regression_model.pth',
            input_size=10,  # Adjust based on your model
            hidden_neurons=512,
            dropout_rate=0.15
        )
        
        # Test with normalization awareness
        test_with_normalized_integers(
            model_path='linear_regression_model.pth',
            input_size=10,
            hidden_neurons=512,
            dropout_rate=0.15
        )
        
        print("\n" + "="*60)
        print("Testing completed successfully!")
        print("="*60)
        
    except FileNotFoundError:
        print("\nError: Model file 'linear_regression_model.pth' not found!")
        print("Please train the model first by running:")
        print("  python src/linear_regression.py")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nMake sure:")
        print("  1. The model file exists")
        print("  2. input_size matches your trained model")
        print("  3. hidden_neurons and dropout_rate match training settings")
