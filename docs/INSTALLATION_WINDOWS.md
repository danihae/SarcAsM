# Installing SarcAsM on Windows

## Recommended: Install via pip

The easiest and most reliable way to use SarcAsM on Windows is through Python:

### Prerequisites
1. Install Python 3.10 or newer from https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. Open Command Prompt or PowerShell

3. Install SarcAsM:
   ```bash
   pip install sarc-asm
   ```

4. Run the application:
   ```bash
   sarcasm
   ```

---

## Why No Standalone Executable for Windows?

Unfortunately, PyTorch (a core dependency) has fundamental incompatibilities with PyInstaller on Windows that cause DLL loading crashes. This is a known issue in the PyInstaller/PyTorch ecosystem and affects many projects.

**The pip installation method works perfectly on Windows** and is actually faster to start up than a bundled executable would be.

---

## Alternative: Anaconda/Miniconda

If you prefer a more isolated environment:

1. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

2. Create a new environment:
   ```bash
   conda create -n sarcasm python=3.12
   conda activate sarcasm
   ```

3. Install SarcAsM:
   ```bash
   pip install sarc-asm
   ```

4. Run:
   ```bash
   sarcasm
   ```

---

## Troubleshooting

If you encounter issues:
- Make sure Python 3.10+ is installed
- Update pip: `python -m pip install --upgrade pip`
- Try in a fresh environment
- Report issues at: https://github.com/danihae/SarcAsM/issues
