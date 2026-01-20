# Make top-level nnunetv2 point to the real nnUNet.nnunetv2

from pathlib import Path

package_path = Path(__file__).parent.parent / "nnUNet" / "nnunetv2"

# Extend __path__ so submodules are found
__path__ = [str(package_path)]
