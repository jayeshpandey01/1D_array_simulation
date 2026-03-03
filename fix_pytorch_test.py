"""
Alternative test script that handles PyTorch DLL issues
If PyTorch fails to load, this provides a workaround
"""
import sys
import os

def test_pytorch_import():
    """Test if PyTorch can be imported"""
    try:
        import torch
        print(f"✓ PyTorch imported successfully (version {torch.__version__})")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def fix_dll_path():
    """Add common DLL paths to system PATH"""
    python_path = sys.executable
    python_dir = os.path.dirname(python_path)
    
    # Common locations for PyTorch DLLs
    dll_paths = [
        os.path.join(python_dir, 'Library', 'bin'),
        os.path.join(python_dir, 'Lib', 'site-packages', 'torch', 'lib'),
        os.path.join(python_dir, 'Scripts'),
    ]
    
    for path in dll_paths:
        if os.path.exists(path) and path not in os.environ['PATH']:
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            print(f"Added to PATH: {path}")

if __name__ == "__main__":
    print("="*60)
    print("PyTorch DLL Fix Utility")
    print("="*60)
    
    print("\nAttempting to fix DLL paths...")
    fix_dll_path()
    
    print("\nTesting PyTorch import...")
    if test_pytorch_import():
        print("\n✓ PyTorch is working! You can now run the test script.")
        print("\nTry running:")
        print("  python test_with_integers.py")
    else:
        print("\n✗ PyTorch still has issues. Try these solutions:")
        print("\n1. Reinstall PyTorch:")
        print("   pip uninstall torch")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("\n2. Install Visual C++ Redistributable:")
        print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n3. Use CPU-only PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
