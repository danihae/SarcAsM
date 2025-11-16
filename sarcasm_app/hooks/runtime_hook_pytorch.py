import os
import sys

def get_binaries():
    """Collect NumPy/SciPy/Pandas/PyTorch DLLs + VC++ runtime - called during Analysis"""
    binaries = []
    if sys.platform == 'win32':
        import glob
        import site
        import shutil
        
        # Try multiple methods to find site-packages
        search_paths = []
        search_paths.append(os.path.join(sys.prefix, 'Lib', 'site-packages'))
        try:
            search_paths.extend(site.getsitepackages())
        except:
            pass
        
        for site_pkg in search_paths:
            # Collect *.libs folders
            for lib_folder in ['numpy.libs', 'scipy.libs', 'pandas.libs']:
                libs_path = os.path.join(site_pkg, lib_folder)
                if os.path.exists(libs_path):
                    dll_files = glob.glob(os.path.join(libs_path, '*.dll'))
                    for dll in dll_files:
                        binaries.append((dll, lib_folder))
                    if dll_files:
                        print(f"[SPEC] Collected {len(dll_files)} DLLs from {libs_path}")
                        break
            
            # Collect PyTorch DLLs from torch/lib
            torch_lib_path = os.path.join(site_pkg, 'torch', 'lib')
            if os.path.exists(torch_lib_path):
                torch_dlls = glob.glob(os.path.join(torch_lib_path, '*.dll'))
                for dll in torch_dlls:
                    binaries.append((dll, 'torch/lib'))
                if torch_dlls:
                    print(f"[SPEC] Collected {len(torch_dlls)} PyTorch DLLs from {torch_lib_path}")
                break
        
        # CRITICAL: Find VC++ Runtime DLLs (c10.dll dependencies)
        vcruntime_dlls = ['vcruntime140.dll', 'msvcp140.dll', 'vcruntime140_1.dll']
        for dll_name in vcruntime_dlls:
            # Use shutil.which() to find DLL in system PATH (most reliable)
            dll_path = shutil.which(dll_name)
            if dll_path:
                binaries.append((dll_path, '.'))
                print(f"[SPEC] Found VC++ runtime: {dll_name}")
            else:
                print(f"[SPEC] WARNING: Missing VC++ runtime: {dll_name}")
    
    if not binaries and sys.platform == 'win32':
        print("[SPEC] WARNING: No DLLs found!")
    else:
        print(f"[SPEC] Total binaries collected: {len(binaries)}")
    
    return binaries
