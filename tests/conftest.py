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
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Also run tests marked slow (minutes of network inference on full stacks)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked slow unless --runslow is given.

    The slow tests run the networks over full-size stacks, which is worth doing
    before a release but makes the everyday suite too slow to run often enough
    to be useful.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow: pass --runslow to include")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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


def _require_test_file(path: Path) -> str:
    """Return ``str(path)`` or skip the test when the data file is unusable.

    A clone without ``git lfs pull`` leaves a ~130-byte pointer file in place of
    each tif, so ``exists()`` alone is not enough: the pointer would be handed
    to tifffile and the test would fail instead of skip.
    """
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        pytest.skip(f"Test data is a Git LFS pointer (run `git lfs pull`): {path}")
    return str(path)


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
    return _require_test_file(test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif")

@pytest.fixture
def structure_timelapse_file_path(test_data_dir):
    """Path to timelapse structure test file."""
    return _require_test_file(test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif")

@pytest.fixture
def structure_single_file_path(test_data_dir):
    """Path to single-image structure test file."""
    return _require_test_file(test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif")


@pytest.fixture(scope="class")
def structure_single_file_path_class(test_data_dir_class):
    """Path to single-image structure test file (class-scoped for plot tests)."""
    return _require_test_file(test_data_dir_class / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif")


def _structure_crop_path():
    """A 512x512 two-frame crop of the time-lapse, carrying its pixel size.

    The full time-lapse is 50 frames of 2000x2000; detecting on all of it takes
    over two minutes. Tests that only need "detection produced masks" get the
    same coverage from this at a fraction of the cost.
    """
    return _get_test_data_dir() / "long_term_2D_ACTN2-citrine_CM" / "structure_crop.tif"


@pytest.fixture
def structure_crop_file_path():
    """Path to the small two-frame crop of the time-lapse."""
    return _require_test_file(_structure_crop_path())


@pytest.fixture(scope="class")
def structure_crop_file_path_class():
    """Path to the small two-frame crop (class-scoped)."""
    return _require_test_file(_structure_crop_path())


@pytest.fixture(scope="session")
def structure_single_image_path(tmp_path_factory):
    """A genuine single-image file, written once per session.

    The single-image fixtures used to point at the 50-frame time-lapse, so the
    "single image" tests neither exercised the single-image path nor ran quickly.
    This writes one frame of the crop with its pixel size preserved.
    """
    source = Path(_require_test_file(_structure_crop_path()))

    import tifffile

    with tifffile.TiffFile(source) as tif:
        frame = tif.series[0].asarray()[0]
        x_res = tif.pages[0].tags["XResolution"].value
    out = tmp_path_factory.mktemp("structure_single") / "single_image.tif"
    tifffile.imwrite(out, frame, imagej=True,
                     resolution=(x_res[0] / x_res[1], x_res[0] / x_res[1]),
                     metadata={"unit": "um"})
    return str(out)


@pytest.fixture
def motion_file_path(test_data_dir):
    """Path to motion test file."""
    return _require_test_file(test_data_dir / "high_speed_single_ACTN2-citrine_CM" / "20kPa.tif")


@pytest.fixture(scope="class")
def motion_file_path_class(test_data_dir_class):
    """Path to motion test file (class-scoped for plot tests)."""
    return _require_test_file(test_data_dir_class / "high_speed_single_ACTN2-citrine_CM" / "20kPa.tif")


@pytest.fixture
def motion_30kPa_file_path(test_data_dir):
    """Path to 30kPa motion test file for domain motion analysis."""
    return _require_test_file(test_data_dir / "high_speed_single_ACTN2-citrine_CM" / "30kPa.tif")


@pytest.fixture(scope="class")
def motion_30kPa_file_path_class(test_data_dir_class):
    """Path to 30kPa motion test file for domain motion analysis (class-scoped)."""
    return _require_test_file(test_data_dir_class / "high_speed_single_ACTN2-citrine_CM" / "30kPa.tif")


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
