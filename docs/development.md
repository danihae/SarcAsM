# Development Guide

This guide is intended for developers who want to contribute to SarcAsM or publish releases.

## Setting Up Development Environment

### Using uv (Recommended) 🚀

1. **Install uv** (if not already installed):
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/danihae/SarcAsM.git
   cd SarcAsM
   ```

3. **Create a development environment**:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate  # On macOS/Linux
   # Or: .venv\Scripts\activate  # On Windows
   ```

4. **Install in development mode with all extras**:
   ```bash
   uv pip install -e ".[dev,test,docs]"
   ```

   This installs the package in editable mode with development, testing, and documentation dependencies. Changes to the code are immediately reflected.

### Using conda (Alternative)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danihae/SarcAsM.git
   cd SarcAsM
   ```

2. **Create a development environment**:
   ```bash
   conda create -n sarcasm-dev python=3.10
   conda activate sarcasm-dev
   ```

3. **Install in development mode with all extras**:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

## Running Tests

SarcAsM uses pytest for testing. To run tests:

```bash
# Install test dependencies (if not already installed)
uv pip install -e ".[test]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_structure.py

# Run a specific test function
pytest tests/test_structure.py::test_function_name

# Run a specific test class
pytest tests/test_structure.py::TestClassName

# Run a specific test method in a class
pytest tests/test_structure.py::TestClassName::test_method_name

# Run tests matching a pattern (keyword expression)
pytest -k "test_sarcomere"  # Runs all tests with "sarcomere" in the name

# Run tests with coverage
pytest --cov=sarcasm tests/

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"

# Stop at first failure (useful when debugging)
pytest -x

# Show local variables in tracebacks (helpful for debugging)
pytest -l

# Verbose output
pytest -v

# Extra verbose (show full diff, etc.)
pytest -vv
```

**Common Workflows:**

```bash
# Debug a single failing test with full output
pytest tests/test_structure.py::test_failing_function -vv -l

# Run all tests in a file except slow ones
pytest tests/test_structure.py -m "not slow"

# Re-run only failed tests from last run
pytest --lf  # "last failed"

# Run failed tests first, then the rest
pytest --ff  # "failed first"
```

Test data should be placed in the `test_data/` directory. If test data is missing, tests will be automatically skipped.

## Code Quality

The project uses several tools for code quality:

* **mypy** for type checking (configuration in `mypy.ini`)
* **ruff** for linting (configuration in `pyproject.toml`)
* **pytest** for testing (configuration in `pytest.ini`)

Install development tools:

```bash
# With uv (recommended)
uv pip install -e ".[dev]"

# With pip
pip install -e ".[dev]"
```

Run linting before committing:

```bash
# Check for issues
ruff check sarcasm/
mypy sarcasm/

# Auto-fix many issues (safe fixes only)
ruff check --fix sarcasm/

# Apply unsafe fixes as well (use with caution)
ruff check --fix --unsafe-fixes sarcasm/

# Format code
ruff format sarcasm/
```

## Building Documentation

Documentation is built using Sphinx. To build locally:

1. **Install documentation dependencies**:
   ```bash
   # With uv (recommended)
   uv pip install -e ".[docs]"
   
   # With pip
   pip install -e ".[docs]"
   ```

2. **Build the docs**:
   ```bash
   cd docs
   make html
   ```

3. **View the documentation**:
   ```bash
   # Open docs/_build/html/index.html in your browser
   open _build/html/index.html  # macOS
   ```

The documentation is automatically built and deployed to ReadTheDocs on each commit to the main branch.

**Note:** The `pyproject.toml` defines all documentation dependencies in the `[project.optional-dependencies.docs]` section.

## Publishing to PyPI

SarcAsM uses GitHub Actions for automated publishing to PyPI. The workflow is triggered by pushing Git tags.

### Automated Publishing via Git Tags (Recommended)

1. **Update version number** in `pyproject.toml`:
   ```toml
   [project]
   name = "sarc-asm"
   version = "X.Y.Z"  # Update this
   ```

2. **Commit and push changes**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

3. **Create and push a Git tag**:

   **For production release to PyPI:**
   ```bash
   git tag vX.Y.Z          # e.g., v1.0.0, v2.1.3
   git push origin vX.Y.Z
   ```

   **For testing release to TestPyPI:**
   ```bash
   git tag vX.Y.Z-test     # e.g., v1.0.0-test, v2.1.3-test
   git push origin vX.Y.Z-test
   ```

4. **GitHub Actions will automatically**:
   - Detect the tag format
   - Build the package with `uv`
   - Publish to PyPI (production tags) or TestPyPI (test tags)
   - Use trusted publishing (no API token needed!)

### Tag Management

```bash
# List all tags
git tag -l

# Delete a tag locally
git tag -d vX.Y.Z

# Delete a tag remotely
git push origin --delete vX.Y.Z

# View tag details
git show vX.Y.Z
```

### Tag Rules

- **Production PyPI**: Tags like `v1.0.0`, `v2.1.3` (no `-test` suffix)
- **TestPyPI**: Tags like `v1.0.0-test`, `v2.1.3-test` (with `-test` suffix)

### Manual Publishing

If you need to publish manually:

```bash
# Install build tools
# With uv (recommended)
uv pip install build twine

# Build the package
uv build

# Upload to PyPI (you'll need PyPI credentials or API token)
uv publish

# Or use twine for more control
python -m build
python -m twine upload dist/*

# Upload to TestPyPI for testing
python -m twine upload --repository testpypi dist/*
```

### GitHub Actions Configuration

The publishing workflows are defined in:
- `.github/workflows/publish.yml` - Production PyPI publishing
- `.github/workflows/publish-test.yml` - TestPyPI publishing

**Key Features:**
- **Trusted Publishing**: Uses OpenID Connect (OIDC) for secure authentication (no API tokens needed!)
- **Tag-based triggers**: Automatically publishes when you push version tags
- **uv for speed**: Uses `uv` for fast package building
- **Environment protection**: Uses GitHub environments for additional security

**Setup Requirements:**

1. **Configure PyPI Trusted Publishing**:
   - Go to PyPI → Your Projects → `sarc-asm` → Publishing
   - Add a new "trusted publisher"
   - Owner: `danihae`
   - Repository: `SarcAsM`
   - Workflow: `publish.yml`
   - Environment: `pypi`

2. **Configure TestPyPI** (optional, for testing):
   - Go to TestPyPI → Your Projects → `sarc-asm` → Publishing
   - Add the same trusted publisher configuration
   - Workflow: `publish-test.yml`

3. **Create GitHub Environment** (optional, for extra protection):
   - Go to GitHub → Settings → Environments → New environment
   - Name it `pypi`
   - Add protection rules if desired (e.g., required reviewers)

## Building Standalone Applications

SarcAsM can be packaged as standalone executables using PyInstaller.

### Windows Executable

On a Windows machine:

```bash
# Install PyInstaller
# With uv (recommended)
uv pip install pyinstaller

# With pip
pip install pyinstaller

# Build the executable
pyinstaller sarcasm.spec

# The executable will be in dist/SarcAsM-vX.Y.Z.exe
```

### macOS Application

On a macOS machine:

```bash
# Install PyInstaller
# With uv (recommended)
uv pip install pyinstaller

# With pip
pip install pyinstaller

# Build the application
pyinstaller sarcasm.spec

# The app bundle will be in dist/SarcAsM-vX.Y.Z.app
```

The build configuration is defined in `sarcasm.spec`.

## Project Structure

Key directories and files:

* **sarcasm/** - Main package source code
  * **structure.py** - Structure analysis (sarcomere detection, Z-bands, etc.)
  * **motion.py** - Motion analysis (tracking, contraction detection)
  * **plots.py** - Plotting functions
  * **export.py** - Data export utilities
  * **structure_modules/** - Modular structure analysis functions

* **sarcasm_app/** - GUI application code
  * **control/** - Application controllers
  * **view/** - UI definitions
  * **model/** - Application models and parameters

* **contraction_net/** - Neural network for contraction detection

* **tests/** - Test suite

* **docs/** - Documentation source files

* **scripts/** - Batch processing scripts

## Version Control

* **Main branch**: `main` - Stable releases only
* **Development**: Create feature branches for new features
* **Releases**: Tagged with version numbers (e.g., `v1.2.3`)

Commit messages should be clear and descriptive. Use conventional commit format when possible:

```
feat: Add new feature
fix: Fix bug in motion tracking
docs: Update documentation
test: Add tests for Z-band analysis
refactor: Refactor sarcomere detection
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License Compliance

This software is licensed under a custom license (see `LICENSE`) and is patent pending (DE 10 2024 112 939.5).

* **Non-commercial use** is free for academic and educational purposes
* **Commercial use** requires a separate license from MBM ScienceBridge GmbH

All contributions must comply with this license structure. By contributing, you agree that your contributions will be licensed under the same terms.

## Contact

For questions about development or contributing:

* Open an issue on GitHub: https://github.com/danihae/SarcAsM/issues
* For licensing inquiries: https://sciencebridge.de/en/
