"""Environment verification script for Fully Self Made OCR.

Reports Python version, installed package versions, and CUDA/GPU availability.
"""
import importlib.metadata
import platform
import sys


REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "Pillow",
    "opencv-contrib-python",
    "pypdfium2",
    "pyclipper",
]


def check_environment():
    print("=" * 60)
    print("Fully Self-Made OCR - Environment Check")
    print("=" * 60)
    py_version = sys.version.split()[0]
    print(f"Python Version : {py_version} ({platform.system()} {platform.machine()})")
    print(f"Executable     : {sys.executable}")

    # Check CUDA / GPU
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available : {cuda_avail}")
        if cuda_avail:
            print(f"Device Name    : {torch.cuda.get_device_name(0)}")
            print(f"Device Count   : {torch.cuda.device_count()}")
        else:
            print("Device         : CPU (default)")
    except ImportError:
        print("CUDA Available : False (torch not installed)")

    print("-" * 60)
    print("Required Dependencies:")
    installed = {}
    for dist in importlib.metadata.distributions():
        try:
            installed[dist.metadata["Name"].lower()] = dist.version
        except Exception:
            continue

    all_present = True
    for pkg in REQUIRED_PACKAGES:
        pkg_lower = pkg.lower()
        version = installed.get(pkg_lower)
        # Check alternate opencv distribution name if opencv-contrib-python not found
        if version is None and pkg_lower == "opencv-contrib-python":
            version = installed.get("opencv-python")
            if version:
                pkg_lower = "opencv-python"
        if version is not None:
            print(f"  [OK]      {pkg:<25} : {version}")
        else:
            print(f"  [MISSING] {pkg:<25} : NOT INSTALLED")
            all_present = False

    print("=" * 60)
    if all_present:
        print("Status: All required dependencies are present.")
        return 0
    else:
        print("Status: Missing dependencies. Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(check_environment())
