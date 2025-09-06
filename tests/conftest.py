import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "test_data"

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

@pytest.fixture
def motion_file_path(test_data_dir):
    """Path to motion test file."""
    file_path = test_data_dir / "high_speed_single_ACTN2-citrine_CM" / "20kPa.tif"    
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
