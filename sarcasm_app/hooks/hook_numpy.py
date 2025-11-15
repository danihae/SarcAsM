# -*- coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
import os
import glob
import sys

binaries = collect_dynamic_libs('numpy')

# NumPy 2.2+ stores DLLs in site-packages/numpy.libs/ (outside package)
if sys.platform == 'win32':
    # Try multiple methods to find site-packages
    site_packages_dirs = []
    
    # Method 1: sys.prefix
    site_packages_dirs.append(os.path.join(sys.prefix, 'Lib', 'site-packages'))
    
    # Method 2: site module
    try:
        import site
        site_packages_dirs.extend(site.getsitepackages())
    except:
        pass
    
    # Method 3: from numpy location
    try:
        import numpy
        numpy_dir = os.path.dirname(numpy.__file__)
        site_packages_dirs.append(os.path.dirname(numpy_dir))
    except:
        pass
    
    # Search all possible locations
    dll_count = 0
    for site_pkg_dir in site_packages_dirs:
        for lib_folder in ['numpy.libs', 'scipy.libs', 'pandas.libs']:
            libs_path = os.path.join(site_pkg_dir, lib_folder)
            if os.path.exists(libs_path):
                dll_files = glob.glob(os.path.join(libs_path, '*.dll'))
                for dll_file in dll_files:
                    binaries.append((dll_file, lib_folder))
                dll_count += len(dll_files)
                print(f"hook-numpy: Collected {len(dll_files)} DLLs from {libs_path}")
    
    if dll_count == 0:
        print(f"hook-numpy ERROR: No DLLs found! Searched: {site_packages_dirs}")

hiddenimports = [
    'numpy.core._multiarray_umath',
    'numpy._core._multiarray_umath',
]
