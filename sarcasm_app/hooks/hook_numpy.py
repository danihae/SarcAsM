# -*- coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
import sys
import os
import glob

binaries = collect_dynamic_libs('numpy')

# Windows: manually collect numpy.libs
if sys.platform == 'win32':
    try:
        import numpy
        numpy_dir = os.path.dirname(numpy.__file__)
        libs_dir = os.path.join(numpy_dir, '.libs')
        
        if os.path.exists(libs_dir):
            dll_files = glob.glob(os.path.join(libs_dir, '*.dll'))
            for dll in dll_files:
                binaries.append((dll, 'numpy.libs'))
            print(f"hook-numpy: Collected {len(dll_files)} DLLs from numpy.libs")
    except Exception as e:
        print(f"hook-numpy ERROR: {e}")

hiddenimports = [
    'numpy.core._multiarray_umath',
    'numpy._core._multiarray_umath',
]
