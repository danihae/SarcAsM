import pytest
import shutil
from pathlib import Path
from typing import List

# Set matplotlib to non-interactive backend before any plotting imports
# This prevents popup windows during tests
import matplotlib
matplotlib.use('Agg')


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep generated *_sarcasm folders after tests complete"
    )


def cleanup_sarcasm_folders(base_paths: List[Path]):
    """
    Remove all *_sarcasm folders adjacent to test data files.
    
    Parameters
    ----------
    base_paths : List[Path]
        List of paths to test data directories to clean up.
    """
    for base_path in base_paths:
        if not base_path.exists():
            continue
        # Find all *_sarcasm directories
        for sarcasm_dir in base_path.glob("*_sarcasm"):
            if sarcasm_dir.is_dir():
                try:
                    shutil.rmtree(sarcasm_dir)
                    print(f"Cleaned up: {sarcasm_dir}")
                except Exception as e:
                    print(f"Warning: Could not remove {sarcasm_dir}: {e}")
        # Also check subdirectories
        for subdir in base_path.iterdir():
            if subdir.is_dir() and not subdir.name.endswith("_sarcasm"):
                for sarcasm_dir in subdir.glob("*_sarcasm"):
                    if sarcasm_dir.is_dir():
                        try:
                            shutil.rmtree(sarcasm_dir)
                            print(f"Cleaned up: {sarcasm_dir}")
                        except Exception as e:
                            print(f"Warning: Could not remove {sarcasm_dir}: {e}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts(request):
    """
    Session-scoped fixture to clean up generated *_sarcasm folders after all tests.
    
    Use --keep-artifacts flag to skip cleanup for debugging.
    """
    yield  # Run all tests first
    
    if request.config.getoption("--keep-artifacts"):
        print("\n--keep-artifacts flag set, skipping cleanup of *_sarcasm folders")
        return
    
    print("\nCleaning up test artifacts...")
    test_data_dir = Path(__file__).parent.parent / "test_data"
    cleanup_sarcasm_folders([
        test_data_dir / "long_term_2D_ACTN2-citrine_CM",
        test_data_dir / "high_speed_single_ACTN2-citrine_CM",
    ])


def _get_test_data_dir():
    """Helper to get test data directory path."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return _get_test_data_dir()


@pytest.fixture(scope="class")
def test_data_dir_class():
    """Path to test data directory (class-scoped)."""
    return _get_test_data_dir()


@pytest.fixture
def structure_metadata_file_path(test_data_dir):
    """Path to structure test file for testing metadata handling."""
    file_path = test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)

@pytest.fixture
def structure_timelapse_file_path(test_data_dir):
    """Path to timelapse structure test file."""
    file_path = test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)

@pytest.fixture
def structure_single_file_path(test_data_dir):
    """Path to single-image structure test file."""
    file_path = test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)


@pytest.fixture(scope="class")
def structure_single_file_path_class(test_data_dir_class):
    """Path to single-image structure test file (class-scoped for plot tests)."""
    file_path = test_data_dir_class / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)


@pytest.fixture
def motion_file_path(test_data_dir):
    """Path to motion test file."""
    file_path = test_data_dir / "high_speed_single_ACTN2-citrine_CM" / "20kPa.tif"    
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)


@pytest.fixture(scope="class")
def motion_file_path_class(test_data_dir_class):
    """Path to motion test file (class-scoped for plot tests)."""
    file_path = test_data_dir_class / "high_speed_single_ACTN2-citrine_CM" / "20kPa.tif"    
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)


@pytest.fixture
def motion_30kPa_file_path(test_data_dir):
    """Path to 30kPa motion test file for domain motion analysis."""
    file_path = test_data_dir / "high_speed_single_ACTN2-citrine_CM" / "30kPa.tif"    
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)


@pytest.fixture(scope="class")
def motion_30kPa_file_path_class(test_data_dir_class):
    """Path to 30kPa motion test file for domain motion analysis (class-scoped)."""
    file_path = test_data_dir_class / "high_speed_single_ACTN2-citrine_CM" / "30kPa.tif"    
    if not file_path.exists():
        pytest.skip(f"Test data not found: {file_path}")
    return str(file_path)


@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Setup and cleanup matplotlib for all tests."""
    import matplotlib.pyplot as plt
    plt.ioff()
    yield
    plt.close('all')


@pytest.fixture
def benchmark_output_dir():
    """Create output directory for benchmark results."""
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    return output_dir
