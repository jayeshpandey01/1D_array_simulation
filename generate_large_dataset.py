"""
Script to generate large waveform datasets for training and testing.
Creates train/test splits and saves them in multiple formats.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import SignalDataset
import matplotlib.pyplot as plt
import os


def visualize_samples(dataset, num_samples=5):
    """Visualize sample waveforms from each class"""
    fig, axes = plt.subplots(5, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get one sample from each class
    for class_idx in range(len(dataset.WAVEFORM_TYPES)):
        # Find first occurrence of this class
        for i in range(len(dataset)):
            if dataset.labels[i] == class_idx:
                signal = dataset.signals[i]
                axes[class_idx].plot(signal)
                axes[class_idx].set_title(f'{dataset.WAVEFORM_TYPES[class_idx]}')
                axes[class_idx].grid(True)
                break
    
    plt.tight_layout()
    plt.savefig('waveform_samples.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved to 'waveform_samples.png'")
    plt.close()


def generate_and_save_datasets(
    train_samples=50000,
    test_samples=10000,
    signal_length=256,
    save_dir='datasets'
):
    """
    Generate large training and testing datasets
    
    Args:
        train_samples: Number of training samples
        test_samples: Number of testing samples
        signal_length: Length of each signal
        save_dir: Directory to save datasets
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("WAVEFORM DATASET GENERATOR")
    print("=" * 60)
    print(f"\nGenerating datasets with:")
    print(f"  - Training samples: {train_samples:,}")
    print(f"  - Testing samples: {test_samples:,}")
    print(f"  - Signal length: {signal_length}")
    print(f"  - Number of classes: 15")
    print(f"  - Save directory: {save_dir}/")
    print("\n" + "=" * 60)
    
    # Generate training dataset
    print("\n[1/4] Generating training dataset...")
    train_dataset = SignalDataset(
        num_samples=train_samples,
        signal_length=signal_length,
        seed=42
    )
    print(f"✓ Training dataset created: {len(train_dataset):,} samples")
    print(f"  Class distribution: {train_dataset.get_class_distribution()}")
    
    # Generate testing dataset
    print("\n[2/4] Generating testing dataset...")
    test_dataset = SignalDataset(
        num_samples=test_samples,
        signal_length=signal_length,
        seed=123
    )
    print(f"✓ Testing dataset created: {len(test_dataset):,} samples")
    print(f"  Class distribution: {test_dataset.get_class_distribution()}")
    
    # Save as PyTorch tensors
    print("\n[3/4] Saving datasets as PyTorch tensors...")
    torch.save({
        'signals': train_dataset.signals,
        'labels': train_dataset.labels,
        'waveform_types': train_dataset.WAVEFORM_TYPES,
        'signal_length': signal_length
    }, os.path.join(save_dir, 'train_dataset.pt'))
    
    torch.save({
        'signals': test_dataset.signals,
        'labels': test_dataset.labels,
        'waveform_types': test_dataset.WAVEFORM_TYPES,
        'signal_length': signal_length
    }, os.path.join(save_dir, 'test_dataset.pt'))
    print(f"✓ PyTorch datasets saved:")
    print(f"  - {save_dir}/train_dataset.pt")
    print(f"  - {save_dir}/test_dataset.pt")
    
    # Save as NumPy arrays
    print("\n[4/4] Saving datasets as NumPy arrays...")
    np.savez_compressed(
        os.path.join(save_dir, 'train_dataset.npz'),
        signals=train_dataset.signals,
        labels=train_dataset.labels,
        waveform_types=train_dataset.WAVEFORM_TYPES
    )
    
    np.savez_compressed(
        os.path.join(save_dir, 'test_dataset.npz'),
        signals=test_dataset.signals,
        labels=test_dataset.labels,
        waveform_types=test_dataset.WAVEFORM_TYPES
    )
    print(f"✓ NumPy datasets saved:")
    print(f"  - {save_dir}/train_dataset.npz")
    print(f"  - {save_dir}/test_dataset.npz")
    
    # Visualize samples
    print("\n[5/5] Creating visualization...")
    visualize_samples(train_dataset)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nTotal samples generated: {train_samples + test_samples:,}")
    print(f"Total storage size: ~{(train_samples + test_samples) * signal_length * 8 / (1024**2):.2f} MB")
    print(f"\nDataset files saved in: {save_dir}/")
    print("\nWaveform types (15 classes):")
    for idx, waveform in enumerate(train_dataset.WAVEFORM_TYPES):
        print(f"  {idx}: {waveform}")
    
    return train_dataset, test_dataset


def load_datasets(save_dir='datasets', format='pytorch'):
    """
    Load previously saved datasets
    
    Args:
        save_dir: Directory containing saved datasets
        format: 'pytorch' or 'numpy'
    
    Returns:
        train_data, test_data
    """
    if format == 'pytorch':
        train_data = torch.load(os.path.join(save_dir, 'train_dataset.pt'))
        test_data = torch.load(os.path.join(save_dir, 'test_dataset.pt'))
    elif format == 'numpy':
        train_data = np.load(os.path.join(save_dir, 'train_dataset.npz'))
        test_data = np.load(os.path.join(save_dir, 'test_dataset.npz'))
    else:
        raise ValueError("Format must be 'pytorch' or 'numpy'")
    
    return train_data, test_data


if __name__ == "__main__":
    # Generate large datasets
    train_dataset, test_dataset = generate_and_save_datasets(
        train_samples=50000,   # 50K training samples
        test_samples=10000,    # 10K testing samples
        signal_length=256,
        save_dir='datasets'
    )
    
    # Example: Create DataLoader for training
    print("\n" + "=" * 60)
    print("EXAMPLE: Creating PyTorch DataLoader")
    print("=" * 60)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    # Get a batch
    signals, labels = next(iter(train_loader))
    print(f"\nBatch shape: {signals.shape}")  # [batch_size, 1, signal_length]
    print(f"Labels shape: {labels.shape}")    # [batch_size]
    print(f"Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    print("\n✓ Dataset ready for model training!")
