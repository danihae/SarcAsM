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

3. **Create a development environment and install**:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate  # On macOS/Linux
   # Or: .venv\Scripts\activate  # On Windows
   
   # Install with all development dependencies and documentation extras
   uv sync --extra docs
   ```

   This installs the package in editable mode with:
   - Core dependencies
   - Development dependencies (from `[dependency-groups]`: pytest, ruff, notebook)
   - Documentation dependencies (from `[project.optional-dependencies]`: sphinx, etc.)

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

3. **Install in development mode**:
   ```bash
   # With pip 25.1+ (supports dependency groups)
   pip install -e . --group dev --extra docs
   
   # With older pip (fallback - requires manual dependency install)
   pip install -e .
   pip install pytest pytest-cov pytest-xdist ruff notebook sphinx sphinx-rtd-theme sphinx-autoapi nbsphinx myst-parser
   ```

## Running Tests

SarcAsM uses pytest for testing. Development dependencies (including pytest) are automatically installed with `uv sync`.

```bash
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

```python
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

### Plot Tests

The test suite includes comprehensive tests for all plotting functions in `sarcasm/plots.py`. These tests verify that plots are generated correctly without errors.

**Test Classes:**

| Test Class | Location | Marker | Description |
|------------|----------|--------|-------------|
| `TestStructureMetadata` | `test_structure.py` | - | Fast metadata tests |
| `TestStructureTimelapseAnalysis` | `test_structure.py` | `slow` | Time-lapse analysis |
| `TestStructureSingleImageAnalysis` | `test_structure.py` | `slow` | Single image analysis |
| `TestStructureErrors` | `test_structure.py` | - | Fast error handling |
| `TestStructureIntegration` | `test_structure.py` | `slow`, `integration` | Full workflow tests |
| `TestStructurePlots` | `test_structure.py` | `slow` | Structure plotting (13 tests) |
| `TestDomainMotionPlots` | `test_structure.py` | `slow` | Domain motion plots (3 tests) |
| `TestMotion` | `test_motion.py` | `slow` | LOI detection and analysis |
| `TestMotionIntegration` | `test_motion.py` | `slow`, `integration` | Full motion workflow |
| `TestMotionPlots` | `test_motion.py` | `slow` | Motion plotting (10 tests) |

**Running Tests by Speed:**

```bash
# Run only fast tests (unit tests, error handling, metadata)
pytest -m "not slow" -v

# Run only slow tests (analysis, detection, plotting)
pytest -m "slow" -v

# Run integration tests only
pytest -m "integration" -v

# Skip both slow and integration tests
pytest -m "not slow and not integration" -v
```

**Running Plot Tests:**

```bash
# Run all structure plot tests
pytest tests/test_structure.py::TestStructurePlots -v

# Run all motion plot tests
pytest tests/test_motion.py::TestMotionPlots -v

# Run domain motion plot tests (slow, uses 30kPa data)
pytest tests/test_structure.py::TestDomainMotionPlots -v

# Run all plot tests together
pytest tests/test_structure.py::TestStructurePlots tests/test_structure.py::TestDomainMotionPlots tests/test_motion.py::TestMotionPlots -v
```

**Test Artifacts:**

Plot tests generate `*_sarcasm/` folders containing analysis results. By default, these are automatically cleaned up after the test session completes.

```bash
# Keep generated folders for debugging
pytest --keep-artifacts -v
```

**Note:** The test suite uses `matplotlib.use('Agg')` to prevent popup windows during testing.

## Code Quality

The project uses several tools for code quality:

* **ruff** for linting and formatting (configuration in `pyproject.toml`)
* **pytest** for testing (configuration in `pytest.ini`)

Development tools are automatically installed with `uv sync` [web:20].

Run linting before committing:

```bash
# Check for issues
ruff check sarcasm/

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
   # With uv (recommended) - includes dev dependencies + docs
   uv sync --extra docs
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

**Note:** The `pyproject.toml` defines documentation dependencies in the `[project.optional-dependencies.docs]` section [web:20].

## Publishing to PyPI

SarcAsM uses GitHub Actions for automated publishing to PyPI. The workflow is triggered by pushing Git tags.

### Automated Publishing via Git Tags (Recommended)

1. **Update version number** in `pyproject.toml`:
   ```
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
# Install build tools (if not already installed)
uv sync --extra app

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
uv sync --extra app

# Build the executable
pyinstaller sarcasm.spec

# The executable will be in dist/SarcAsM-vX.Y.Z.exe
```

### macOS Application

On a macOS machine:

```bash
# Install PyInstaller
uv sync --extra app

# Build the application
pyinstaller sarcasm.spec

# The app bundle will be in dist/SarcAsM-vX.Y.Z.app
```

**Note for macOS users**: The built app is not code-signed. Users will need to bypass the security warning on first launch by right-clicking the app and selecting "Open". See [INSTALLATION_MACOS.md](../INSTALLATION_MACOS.md) for detailed user instructions.

To test the built app locally:
```bash
# Remove quarantine attribute to bypass security warning
xattr -cr dist/SarcAsM-v*.app

# Run the app
open dist/SarcAsM-v*.app
```

**Note**: Code signing requires an Apple Developer account ($99/year), which is not feasible for academic/scientific software.

The build configuration is defined in `sarcasm.spec`.

## Logging

SarcAsM uses Python's standard `logging` module throughout the codebase. Each module has its own logger:

```python
import logging
logger = logging.getLogger(__name__)
```

### Log Levels Used

| Level | Usage |
|-------|-------|
| `DEBUG` | Diagnostic info, fallback attempts, internal state |
| `INFO` | Major processing steps, analysis progress |
| `WARNING` | Non-critical issues, suboptimal parameters |
| `ERROR` | Failures that prevent operation |

### Controlling Log Level

Users can set the log level when initializing `Structure` or `Motion` objects:

```python
from sarcasm import Structure

sarc = Structure('file.tif', log_level='DEBUG')  # Verbose
sarc = Structure('file.tif', log_level='WARNING')  # Quiet
```

### GUI Integration

The GUI application automatically displays log messages in the message area at the bottom of the window. Log messages are color-coded:

- **Gray**: DEBUG
- **White**: INFO  
- **Yellow**: WARNING
- **Red**: ERROR

The GUI handler is in `sarcasm_app/control/logging_handler.py`.

### Adding Logging to New Code

When adding new functionality:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting process...")
    try:
        # ... code ...
        logger.debug(f"Intermediate value: {value}")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

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

## Contact

For questions about development or contributing:

* Open an issue on GitHub: https://github.com/danihae/SarcAsM/issues
* For licensing inquiries: https://sciencebridge.de/en/
