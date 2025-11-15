# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys
import os

# Get SarcAsM version
try:
    from sarcasm import __version__ as version
except ImportError:
    version = '0.0.0-import-error'

# Dynamic platform-aware naming
appname = f"SarcAsM-v{version}"

# Collect data files - be comprehensive, not selective
napari_data = collect_data_files('napari', include_py_files=False)
napari_data += collect_data_files('napari_builtins', include_py_files=False)
vispy_data = collect_data_files('vispy', include_py_files=False)
model_data = [('sarcasm/models', 'sarcasm/models')]

# Only exclude things that are definitely NOT needed
excludes = [
    # Exclude test frameworks and dev tools (but NOT unittest - torch needs it!)
    'pytest', '_pytest', 'nose', 'hypothesis',
    'IPython', 'jupyter', 'notebook', 'nbconvert', 'nbformat',
    'tkinter', 'tcl', 'tk', '_tkinter',
    # Exclude test submodules (but NOT numpy.testing - scipy needs it!)
    'scipy.tests',
    'matplotlib.tests',
    'PIL.ImageQt',
]

# Platform-specific excludes for Windows to avoid torch import issues during analysis
if sys.platform == 'win32':
    # Don't let PyInstaller try to import torch during analysis - causes access violations
    # The hook will collect it properly instead
    pass  # We'll handle torch via hook, not via excludes

a = Analysis(
    ['sarcasm_app/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=napari_data + vispy_data + model_data,
    hiddenimports=[
        'napari',
        'napari._qt',
        'napari.plugins',
        'napari_builtins',
        'vispy',
        'vispy.glsl',
        'vispy.app',
        'vispy.app.backends',
        'vispy.app.backends._pyqt5',
        'PyQt5.QtOpenGL',
        'freetype',
        'PyQt5.sip',
        # Torch imports for Windows compatibility
        'torch',
        'torch._C',
        'torch._VF',
    ] + collect_submodules('sarcasm_app') + collect_submodules('vispy'),
    hookspath=['sarcasm_app/hooks'],
    hooksconfig={},
    runtime_hooks=['sarcasm_app/hooks/runtime_hook_matplotlib.py'],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=appname,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Faster startup without compression
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='sarcasm_app/icons/sarcasm.ico',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name=f'{appname}.app',
        icon='sarcasm_app/icons/sarcasm.icns',
        bundle_identifier='de.umg.sarcasm',
        info_plist={
            'CFBundleName': 'SarcAsM',
            'CFBundleDisplayName': 'SarcAsM',
            'CFBundleShortVersionString': version,
            'CFBundleVersion': version,
            'NSHighResolutionCapable': 'True',
            'LSUIElement': 'False',
            'NSRequiresAquaSystemAppearance': 'False',
        }
    )
else:
    pass
