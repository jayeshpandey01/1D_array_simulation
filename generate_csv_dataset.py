"""
Generate large waveform datasets and save as CSV files.
Creates separate train and test CSV files with all 17 waveform types.
"""

import numpy as np
import pandas as pd
from src.dataset import SignalDataset
import os
from tqdm import tqdm
import time


def generate_csv_datasets(
    train_samples=50000,
    test_samples=10000,
    signal_length=256,
    save_dir='datasets',
    chunk_size=5000
):
    """
    Generate large datasets and save as CSV files
    
    Args:
        train_samples: Number of training samples
        test_samples: Number of testing samples
        signal_length: Length of each signal
        save_dir: Directory to save CSV files
        chunk_size: Number of samples to generate at a time (for memory efficiency)
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("WAVEFORM DATASET CSV GENERATOR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Testing samples: {test_samples:,}")
    print(f"  Signal length: {signal_length}")
    print(f"  Number of classes: 15")
    print(f"  Save directory: {save_dir}/")
    print("=" * 70)
    
    # Generate training dataset
    print("\n[1/2] Generating TRAINING dataset...")
    train_path = os.path.join(save_dir, 'train_waveforms.csv')
    generate_csv_in_chunks(
        num_samples=train_samples,
        signal_length=signal_length,
        output_path=train_path,
        chunk_size=chunk_size,
        seed=42
    )
    
    # Generate testing dataset
    print("\n[2/2] Generating TESTING dataset...")
    test_path = os.path.join(save_dir, 'test_waveforms.csv')
    generate_csv_in_chunks(
        num_samples=test_samples,
        signal_length=signal_length,
        output_path=test_path,
        chunk_size=chunk_size,
        seed=123
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 70)
    
    train_size_mb = os.path.getsize(train_path) / (1024 ** 2)
    test_size_mb = os.path.getsize(test_path) / (1024 ** 2)
    
    print(f"\nFiles created:")
    print(f"  Training: {train_path}")
    print(f"    - Samples: {train_samples:,}")
    print(f"    - Size: {train_size_mb:.2f} MB")
    print(f"\n  Testing: {test_path}")
    print(f"    - Samples: {test_samples:,}")
    print(f"    - Size: {test_size_mb:.2f} MB")
    print(f"\n  Total size: {train_size_mb + test_size_mb:.2f} MB")
    
    print("\nWaveform types (15 classes):")
    for idx, waveform in enumerate(SignalDataset.WAVEFORM_TYPES):
        print(f"  {idx}: {waveform}")
    
    # Show preview
    print("\n" + "=" * 70)
    print("DATASET PREVIEW")
    print("=" * 70)
    df_preview = pd.read_csv(train_path, nrows=5)
    print(f"\nFirst 5 rows of training data:")
    print(df_preview.to_string())
    
    print(f"\nColumns: {list(df_preview.columns)}")
    print(f"  - signal_0 to signal_{signal_length-1}: Signal values")
    print(f"  - label: Numeric class label (0-14)")
    print(f"  - label_name: Human-readable waveform type")


def generate_csv_in_chunks(num_samples, signal_length, output_path, chunk_size=5000, seed=None):
    """
    Generate dataset in chunks and write to CSV to manage memory
    """
    if seed is not None:
        np.random.seed(seed)
    
    start_time = time.time()
    first_chunk = True
    
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    print(f"  Generating {num_samples:,} samples in {num_chunks} chunks...")
    
    for chunk_idx in range(num_chunks):
        # Calculate samples for this chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_samples = end_idx - start_idx
        
        # Generate chunk
        chunk_dataset = SignalDataset(
            num_samples=chunk_samples,
            signal_length=signal_length,
            seed=None  # Don't reset seed for each chunk
        )
        
        # Convert to DataFrame
        df_chunk = signals_to_dataframe(
            chunk_dataset.signals,
            chunk_dataset.labels,
            chunk_dataset.WAVEFORM_TYPES
        )
        
        # Write to CSV
        if first_chunk:
            df_chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            df_chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        # Progress update
        progress = (chunk_idx + 1) / num_chunks * 100
        elapsed = time.time() - start_time
        eta = elapsed / (chunk_idx + 1) * (num_chunks - chunk_idx - 1)
        
        print(f"  Progress: {progress:.1f}% | Chunk {chunk_idx+1}/{num_chunks} | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Complete! Generated {num_samples:,} samples in {elapsed:.1f}s")


def signals_to_dataframe(signals, labels, waveform_types):
    """
    Convert signals and labels to a pandas DataFrame
    """
    # Create column names for signal values
    signal_columns = [f'signal_{i}' for i in range(signals.shape[1])]
    
    # Create DataFrame from signals
    df = pd.DataFrame(signals, columns=signal_columns)
    
    # Add labels
    df['label'] = labels
    df['label_name'] = [waveform_types[label] for label in labels]
    
    return df


def load_csv_dataset(filepath, return_numpy=True):
    """
    Load a CSV dataset
    
    Args:
        filepath: Path to CSV file
        return_numpy: If True, return numpy arrays; if False, return DataFrame
    
    Returns:
        If return_numpy=True: (signals, labels, label_names)
        If return_numpy=False: DataFrame
    """
    df = pd.read_csv(filepath)
    
    if return_numpy:
        # Extract signal columns
        signal_cols = [col for col in df.columns if col.startswith('signal_')]
        signals = df[signal_cols].values
        labels = df['label'].values
        label_names = df['label_name'].values
        return signals, labels, label_names
    else:
        return df


def analyze_dataset(filepath):
    """
    Analyze and print statistics about a CSV dataset
    """
    print(f"\nAnalyzing: {filepath}")
    print("-" * 70)
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Basic info
    print(f"Total samples: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Class distribution
    print("\nClass distribution:")
    class_dist = df['label_name'].value_counts().sort_index()
    for label_name, count in class_dist.items():
        percentage = count / len(df) * 100
        print(f"  {label_name:25s}: {count:6,} ({percentage:5.2f}%)")
    
    # Signal statistics
    signal_cols = [col for col in df.columns if col.startswith('signal_')]
    signals = df[signal_cols].values
    
    print(f"\nSignal statistics:")
    print(f"  Signal length: {len(signal_cols)}")
    print(f"  Min value: {signals.min():.4f}")
    print(f"  Max value: {signals.max():.4f}")
    print(f"  Mean value: {signals.mean():.4f}")
    print(f"  Std deviation: {signals.std():.4f}")
    
    # File size
    file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
    print(f"\nFile size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    # Generate large CSV datasets
    generate_csv_datasets(
        train_samples=50000,    # 50K training samples
        test_samples=10000,     # 10K testing samples
        signal_length=256,      # 256 time steps per signal
        save_dir='datasets',
        chunk_size=5000         # Generate 5000 samples at a time
    )
    
    # Analyze the generated datasets
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    analyze_dataset('datasets/train_waveforms.csv')
    analyze_dataset('datasets/test_waveforms.csv')
    
    # Example: Load and use the dataset
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    
    print("\nTo load the CSV dataset:")
    print("  signals, labels, label_names = load_csv_dataset('datasets/train_waveforms.csv')")
    print("  print(f'Signals shape: {signals.shape}')")
    print("  print(f'Labels shape: {labels.shape}')")
    
    print("\nOr load as pandas DataFrame:")
    print("  df = pd.read_csv('datasets/train_waveforms.csv')")
    print("  signal_cols = [col for col in df.columns if col.startswith('signal_')]")
    print("  X = df[signal_cols].values")
    print("  y = df['label'].values")
