from PyInstaller.utils.hooks import collect_dynamic_libs
import os
import glob
import site
import sys

binaries = collect_dynamic_libs('numpy')

# NumPy 2.2+ stores DLLs in site-packages/numpy.libs/ (outside package)
if sys.platform == 'win32':
    for site_pkg_dir in site.getsitepackages():
        for lib_folder in ['numpy.libs', 'scipy.libs', 'pandas.libs']:
            libs_path = os.path.join(site_pkg_dir, lib_folder)
            if os.path.exists(libs_path):
                for dll_file in glob.glob(os.path.join(libs_path, '*.dll')):
                    binaries.append((dll_file, lib_folder))
                print(f"hook-numpy: Found {lib_folder} with {len(glob.glob(os.path.join(libs_path, '*.dll')))} DLLs")

hiddenimports = [
    'numpy.core._multiarray_umath',
    'numpy._core._multiarray_umath',
]
