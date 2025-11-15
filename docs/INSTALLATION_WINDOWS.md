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

## Experimental standalone ZIP build

Every tagged release now ships a Windows ZIP archive (e.g. `SarcAsM-vX.Y.Z-windows.zip`) that contains a frozen `.exe`. To work around PyTorch's DLL loader crash we package the app with PyInstaller's `--no-isolate` mode, which keeps the analysis inside a single Python process during the build.

### How to use the ZIP build
1. Download the latest ZIP from the [GitHub Releases page](https://github.com/danihae/SarcAsM/releases).
2. Right-click the ZIP in Explorer, open **Properties**, and check **Unblock** if it exists.
3. Extract the archive and double-click the `SarcAsM-*.exe` inside `dist`.
4. If SmartScreen warns about an unrecognized app, choose *More info* → *Run anyway*.

> ⚠️ The standalone is still considered experimental. If Windows Defender or driver policies prevent the app from launching, fall back to the pip installation above, which remains the most reliable option.

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
