"""
Runtime hook for matplotlib to speed up startup.
This hook sets a persistent matplotlib config directory to avoid
rebuilding the font cache on every app launch.
"""
import os
import sys

# Set matplotlib config to a writable location
if getattr(sys, 'frozen', False):
    # Running in PyInstaller bundle
    mpl_config = os.path.join(os.path.expanduser('~'), '.sarcasm_mpl')
    os.environ['MPLCONFIGDIR'] = mpl_config
    
    # Disable matplotlib's font manager findfont warnings (speeds up import)
    os.environ['MPLBACKEND'] = 'Agg'  # Use non-GUI backend by default
    
    # Create directory if it doesn't exist
    if not os.path.exists(mpl_config):
        os.makedirs(mpl_config, exist_ok=True)
