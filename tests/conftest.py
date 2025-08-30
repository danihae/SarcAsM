import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "test_data"

@pytest.fixture
def structure_metadata_filepath(test_data_dir):
    """Path to structure test file for testing metadata handling."""
    filepath = test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not filepath.exists():
        pytest.skip(f"Test data not found: {filepath}")
    return str(filepath)

@pytest.fixture
def structure_timelapse_filepath(test_data_dir):
    """Path to timelapse structure test file."""
    filepath = test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not filepath.exists():
        pytest.skip(f"Test data not found: {filepath}")
    return str(filepath)

@pytest.fixture
def structure_single_filepath(test_data_dir):
    """Path to single-image structure test file."""
    filepath = test_data_dir / "long_term_2D_ACTN2-citrine_CM" / "20211115_ACTN2_CMs_96well_control_12days.tif"
    if not filepath.exists():
        pytest.skip(f"Test data not found: {filepath}")
    return str(filepath)

@pytest.fixture
def motion_filepath(test_data_dir):
    """Path to motion test file."""
    filepath = test_data_dir / "high_speed_single_ACTN2-citrine_CM" / "20kPa.tif"    
    if not filepath.exists():
        pytest.skip(f"Test data not found: {filepath}")
    return str(filepath)

@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Setup and cleanup matplotlib for all tests."""
    import matplotlib.pyplot as plt
    plt.ioff()
    yield
    plt.close('all')
