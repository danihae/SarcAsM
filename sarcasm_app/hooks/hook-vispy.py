"""
PyInstaller hook for vispy to ensure PyQt5 backend is properly included.
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

# Collect ALL vispy submodules to ensure backend is properly packaged
hiddenimports = collect_submodules('vispy')

# Explicitly ensure critical PyQt5 backend components
hiddenimports += [
    'PyQt5.QtOpenGL',
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
]

# Collect vispy data files and metadata
datas = collect_data_files('vispy')
datas += copy_metadata('vispy')
