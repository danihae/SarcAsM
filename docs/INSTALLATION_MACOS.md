# Installing SarcAsM on macOS

## Download

Download the latest `SarcAsM-vX.X.X-macos.dmg` from the [Releases page](https://github.com/danihae/SarcAsM/releases).

## Installation

### Step 1: Install the App
1. Open the downloaded DMG file
2. Drag SarcAsM.app to your Applications folder

### Step 2: Remove Security Block (One Command)

Because this is free academic software without Apple's $99/year code-signing, macOS blocks it. 

**Copy and paste this into Terminal:**

```bash
xattr -cr /Applications/SarcAsM-v*.app
```

**How to open Terminal:**
- Press `Command + Space` to open Spotlight
- Type `Terminal` and press Enter
- Paste the command above and press Enter

That's it! Now you can open SarcAsM normally from your Applications folder.

---

## What This Command Does

The `xattr -cr` command removes the "downloaded from internet" quarantine flag that macOS adds to files downloaded from the web. It's completely safe and makes the app behave exactly like if you'd built it yourself on your Mac.

## Why Is This Necessary?

This app is free, open-source scientific software. Apple requires a $99/year Developer certificate to avoid this security step - which isn't practical for academic research tools.

**Many scientific applications work the same way:**
- ImageJ / Fiji
- Many R packages and tools
- Python-based scientific software
- MATLAB community toolboxes

---

## Uninstallation

Simply drag SarcAsM.app from your Applications folder to the Trash.

---

## Alternative: Install via Python

If you prefer, you can install using pip:
```bash
pip install sarc-asm
sarcasm  # Run the application
```
