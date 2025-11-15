"""
PyInstaller hook for torch on Windows
Ensures all torch DLLs and dependencies are properly included
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, get_package_paths
import os

# Collect all torch binaries and DLLs
binaries = collect_dynamic_libs('torch')

# Collect torch data files
datas = collect_data_files('torch', include_py_files=False)

# Ensure critical torch modules are imported
hiddenimports = [
    'torch._C',
    'torch._VF',
    'torch.cuda',
    'torch.version',
]

# On Windows, we need to ensure torch DLLs are in the right location
import sys
if sys.platform == 'win32':
    # Get torch library path
    pkg_base, pkg_dir = get_package_paths('torch')
    torch_lib_dir = os.path.join(pkg_dir, 'lib')
    
    # Collect all DLLs from torch/lib directory
    if os.path.exists(torch_lib_dir):
        for filename in os.listdir(torch_lib_dir):
            if filename.endswith('.dll'):
                dll_path = os.path.join(torch_lib_dir, filename)
                binaries.append((dll_path, 'torch/lib'))
