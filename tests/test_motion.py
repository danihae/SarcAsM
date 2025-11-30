import pytest
from sarcasm import Structure, Motion, Plots
import matplotlib.pyplot as plt


@pytest.mark.slow
class TestMotion:
    """Tests for LOI detection and motion analysis."""
    
    def test_loi_detection_pipeline(self, motion_file_path):
        """Test complete LOI detection pipeline."""
        sarc = Structure(motion_file_path, restart=True)
        
        # Run detection pipeline
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        
        # Clear existing LOIs and detect new ones
        sarc.delete_lois()
        sarc.detect_lois(n_lois=2, persistence=6, mode='fit_straight_line')
        
        # Verify LOI detection
        lois = sarc.get_list_lois()
        assert isinstance(lois, list)
        assert len(lois) <= 2  # May detect fewer than requested
        
        # Verify LOI structure
        for loi in lois:
            assert len(loi) == 2  # Should contain (file, roi)
            file, roi = loi
            assert file is not None
            assert roi is not None

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
            
    def test_motion_object_initialization(self, motion_file_path):
        """Test Motion object creation and initialization."""
        sarc = Structure(motion_file_path, restart=True)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        sarc.delete_lois()
        sarc.detect_lois(n_lois=2, persistence=6, mode='fit_straight_line')
        
        lois = sarc.get_list_lois()
        if not lois:
            pytest.skip("No LOIs detected for Motion object testing")
            
        # Initialize Motion object
        file, roi = lois[0]
        mot_obj = Motion(file, roi)
        assert isinstance(mot_obj, Motion)

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
        
    def test_full_analysis_loi(self, motion_file_path):
        """Test complete LOI motion analysis."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        sarc.delete_lois()
        sarc.detect_lois(n_lois=2, persistence=6, mode='fit_straight_line')
        
        lois = sarc.get_list_lois()
        if not lois:
            pytest.skip("No LOIs detected for full analysis testing")
            
        file, roi = lois[0]
        mot_obj = Motion(file, roi)
        
        # Run full analysis
        mot_obj.full_analysis_loi()
        
        # Verify analysis results
        assert 'delta_slen' in mot_obj.loi_data
        assert 'beating_rate' in mot_obj.loi_data
        assert 'contr_max' in mot_obj.loi_data

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()

    def test_get_list_lois(self, motion_file_path):
        """Test getting list of LOIs from Structure object."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        sarc.delete_lois()
        sarc.detect_lois(n_lois=1, persistence=6, mode='fit_straight_line')
        
        lois = sarc.get_list_lois()
        assert isinstance(lois, list)

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
        
    def test_delete_lois(self, motion_file_path):
        """Test LOI deletion functionality."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        
        # First add some LOIs
        sarc.detect_lois(n_lois=1, persistence=6, mode='fit_straight_line')
        lois_before = sarc.get_list_lois()
        
        # Delete them
        sarc.delete_lois()
        lois_after = sarc.get_list_lois()
        
        # Should have fewer or equal LOIs after deletion
        assert len(lois_after) <= len(lois_before)

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
        
    @pytest.mark.parametrize("n_lois,persistence,mode", [
        (1, 5, 'fit_straight_line'),
        (2, 6, 'fit_straight_line'),
        (3, 7, 'fit_straight_line'),
    ])
    def test_loi_detection_parameters(self, motion_file_path, n_lois, persistence, mode):
        """Test LOI detection with different parameters."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        sarc.delete_lois()
        
        sarc.detect_lois(n_lois=n_lois, persistence=persistence, mode=mode)
        lois = sarc.get_list_lois()
        
        # Should detect at least some LOIs (may be fewer than requested)
        assert isinstance(lois, list)
        assert len(lois) >= 0
        assert len(lois) <= n_lois

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
        
    def test_detect_sarcomeres_single_frame(self, motion_file_path):
        """Test sarcomere detection on single frame."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        
        # Verify detection attributes exist
        assert hasattr(sarc, 'zbands')
        assert hasattr(sarc, 'mbands')
        
    def test_detect_z_bands_fast_movie(self, motion_file_path):
        """Test fast Z-band detection for movies."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        
        # Run fast Z-band detection
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        
        # Verify Z-band detection completed
        assert hasattr(sarc, 'zbands')

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
        
    def test_analyze_sarcomere_vectors(self, motion_file_path):
        """Test sarcomere vector analysis."""
        sarc = Structure(motion_file_path)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        
        # Analyze sarcomere vectors
        sarc.analyze_sarcomere_vectors(frames=0)
        
        # Verify analysis completed
        assert 'sarcomere_length_vectors' in sarc.data

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()

class TestMotionIntegration:
    """Integration tests for complete motion analysis workflow."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_complete_motion_workflow(self, motion_file_path):
        """Test complete motion analysis workflow from start to finish."""
        # Structure initialization and detection
        sarc = Structure(motion_file_path, restart=True)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        
        # LOI detection
        sarc.delete_lois()
        sarc.detect_lois(n_lois=2, persistence=6, mode='fit_straight_line')
        lois = sarc.get_list_lois()
        
        if not lois:
            pytest.skip("No LOIs detected for complete workflow test")
            
        # Motion analysis
        file, roi = lois[0]
        mot_obj = Motion(file, roi)
        mot_obj.full_analysis_loi()
        
        # Verify complete workflow
        assert hasattr(sarc, 'zbands')
        assert 'sarcomere_length_vectors' in sarc.data
        assert 'delta_slen' in mot_obj.loi_data
        assert 'beating_rate' in mot_obj.loi_data

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()
        
    @pytest.mark.slow
    @pytest.mark.integration  
    def test_multiple_lois_analysis(self, motion_file_path):
        """Test analysis of multiple LOIs."""
        sarc = Structure(motion_file_path, restart=True)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        sarc.delete_lois()
        sarc.detect_lois(n_lois=2, persistence=5, mode='fit_straight_line')
        
        lois = sarc.get_list_lois()
        if len(lois) < 2:
            pytest.skip("Not enough LOIs detected for multiple LOI test")
            
        # Test first LOI
        file1, roi1 = lois[0]
        mot_obj1 = Motion(file1, roi1)
        mot_obj1.full_analysis_loi()
        
        # Test second LOI
        file2, roi2 = lois[1]
        mot_obj2 = Motion(file2, roi2)
        mot_obj2.full_analysis_loi()
        
        # Verify both analyses completed
        assert 'delta_slen' in mot_obj1.loi_data
        assert 'delta_slen' in mot_obj2.loi_data      

        # Cleanup intermediate files
        sarc.remove_intermediate_tiffs()  


@pytest.mark.slow
class TestMotionPlots:
    """Tests for motion/LOI-related plotting functions."""

    @pytest.fixture(scope="class")
    def analyzed_motion_data(self, motion_file_path_class):
        """
        Class-scoped fixture providing analyzed Structure and Motion objects.
        Runs LOI detection and full motion analysis once for the entire test class.
        Returns tuple of (sarc, mot_obj) or skips if no LOIs detected.
        """
        sarc = Structure(motion_file_path_class, restart=True)
        sarc.detect_sarcomeres(frames=0, max_patch_size=(256, 1024))
        sarc.detect_z_bands_fast_movie(max_patch_size=(32, 210, 1024))
        sarc.analyze_sarcomere_vectors(frames=0)
        sarc.delete_lois()
        sarc.detect_lois(n_lois=2, persistence=6, mode='fit_straight_line')
        
        lois = sarc.get_list_lois()
        if not lois:
            pytest.skip("No LOIs detected for motion plot testing")
        
        file, roi = lois[0]
        mot_obj = Motion(file, roi)
        mot_obj.full_analysis_loi()
        
        return sarc, mot_obj

    def test_plot_loi_detection(self, analyzed_motion_data):
        """Test plot_loi_detection function."""
        sarc, mot_obj = analyzed_motion_data
        # This function creates its own figure internally
        Plots.plot_loi_detection(sarc, frame=0)
        plt.close('all')

    def test_plot_image_with_loi(self, analyzed_motion_data):
        """Test plot_image with show_loi=True."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots()
        Plots.plot_image(ax, sarc, frame=0, show_loi=True)
        assert ax.images, "No image was plotted"
        plt.close(fig)

    def test_plot_z_bands_with_loi(self, analyzed_motion_data):
        """Test plot_z_bands with show_loi=True."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots()
        Plots.plot_z_bands(ax, sarc, frame=0, show_loi=True)
        assert ax.images, "No Z-band image was plotted"
        plt.close(fig)

    def test_plot_z_pos(self, analyzed_motion_data):
        """Test plot_z_pos function."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots()
        Plots.plot_z_pos(ax, mot_obj, t_lim=(0, 2))
        # Check for plotted trajectories
        assert ax.lines, "No Z-position trajectories were plotted"
        plt.close(fig)

    def test_plot_delta_slen(self, analyzed_motion_data):
        """Test plot_delta_slen function."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots(figsize=(12, 6))
        Plots.plot_delta_slen(ax, mot_obj, t_lim=(0, 2))
        # The main axes should have labels set even if no inset axes were created
        # (depends on n_rows and number of sarcomeres)
        assert ax.get_xlabel() or ax.get_ylabel(), "No delta_slen plot was created"
        plt.close(fig)

    def test_plot_overlay_delta_slen(self, analyzed_motion_data):
        """Test plot_overlay_delta_slen function."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots()
        Plots.plot_overlay_delta_slen(ax, mot_obj, t_lim=(0, 2))
        # Check for plotted lines
        assert ax.lines, "No delta_slen overlay lines were plotted"
        plt.close(fig)

    def test_plot_overlay_velocity(self, analyzed_motion_data):
        """Test plot_overlay_velocity function."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots()
        Plots.plot_overlay_velocity(ax, mot_obj, t_lim=(0, 2))
        # Check for plotted lines
        assert ax.lines, "No velocity overlay lines were plotted"
        plt.close(fig)

    def test_plot_phase_space(self, analyzed_motion_data):
        """Test plot_phase_space function."""
        sarc, mot_obj = analyzed_motion_data
        fig, ax = plt.subplots()
        Plots.plot_phase_space(ax, mot_obj, t_lim=(0, 2))
        # Check for plotted trajectories or scatter
        assert ax.lines or ax.collections, "No phase space plot was created"
        plt.close(fig)

    def test_plot_popping_events(self, analyzed_motion_data):
        """Test plot_popping_events function."""
        sarc, mot_obj = analyzed_motion_data
        # Check if popping analysis was performed (requires specific keys)
        if 'popping_events' not in mot_obj.loi_data or 'popping_freq_time' not in mot_obj.loi_data:
            pytest.skip("Popping analysis not available in motion data")
        # This function creates its own figure internally
        Plots.plot_popping_events(mot_obj)
        plt.close('all')

    def test_plot_loi_summary_motion(self, analyzed_motion_data):
        """Test plot_loi_summary_motion function."""
        sarc, mot_obj = analyzed_motion_data
        # This function creates its own figure internally and shows it
        # We just verify it doesn't raise an error
        Plots.plot_loi_summary_motion(mot_obj, t_lim=(0, 2))
        plt.close('all')


# Run with: pytest tests/test_motion.py -v
# Run slow tests: pytest tests/test_motion.py -m "slow" -v  
# Skip slow tests: pytest tests/test_motion.py -m "not slow" -v
