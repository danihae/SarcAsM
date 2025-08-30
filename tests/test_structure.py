import pytest
import os
from sarcasm import Structure


class TestStructureMetadata:
    """Test metadata functionality on one fixed file."""

    def test_initialization_auto_pixelsize(self, structure_metadata_filepath):
        """Test basic Structure initialization with auto pixel size."""
        sarc = Structure(structure_metadata_filepath, restart=True)
        assert isinstance(sarc, Structure)
        assert sarc.filepath is not None
        assert sarc.metadata.pixelsize is not None
        
    def test_initialization_manual_pixelsize(self, structure_metadata_filepath):
        """Test Structure initialization with manual pixel size."""
        sarc = Structure(structure_metadata_filepath, restart=True, pixelsize=0.1)
        assert isinstance(sarc, Structure)
        assert sarc.metadata.pixelsize == 0.1
        
    def test_initialization_with_metadata(self, structure_metadata_filepath):
        """Test Structure initialization with additional metadata."""
        sarc = Structure(structure_metadata_filepath, cell_line='WT', treatment='control', restart=True)
        assert isinstance(sarc, Structure)
        assert sarc.metadata.user_info['cell_line'] == 'WT'
        assert sarc.metadata.user_info['treatment'] == 'control'
        
    def test_multiple_metadata_entries(self, structure_metadata_filepath):
        """Test Structure with multiple metadata entries."""
        metadata = {
            'experiment_date': '2025-08-29',
            'concentration': '30kPa',
            'cell_type': 'cardiomyocyte',
            'researcher': 'Daniel'
        }
        sarc = Structure(structure_metadata_filepath, **metadata, restart=True)
        
        for key, value in metadata.items():
            assert sarc.metadata.user_info[key] == value
        
    def test_structure_metadata_properties(self, structure_metadata_filepath):
        """Test that metadata is properly initialized."""
        sarc = Structure(structure_metadata_filepath, restart=False)
        
        # Check core metadata properties exist
        assert hasattr(sarc, 'metadata')
        assert hasattr(sarc.metadata, 'file_name')
        assert hasattr(sarc.metadata, 'file_path')
        assert hasattr(sarc.metadata, 'sarcasm_version')
        assert hasattr(sarc.metadata, 'timestamp_analysis')
        
    def test_filepath_storage(self, structure_metadata_filepath):
        """Test that file path is correctly stored."""
        sarc = Structure(structure_metadata_filepath, restart=False)
        assert structure_metadata_filepath in sarc.filepath or structure_metadata_filepath == sarc.filepath


class TestStructureTimelapseAnalysis:
    """Test analysis pipeline on time-lapse files."""

    @pytest.mark.slow
    def test_timelapse_sarcomere_detection(self, structure_timelapse_filepath):
        """Test sarcomere detection on time-lapse."""
        sarc = Structure(structure_timelapse_filepath, restart=False)
        sarc.detect_sarcomeres(max_patch_size=(1024, 1024))
        
        # Verify detection attributes exist
        assert hasattr(sarc, 'zbands')
        assert hasattr(sarc, 'mbands')
        assert hasattr(sarc, 'cell_mask')
        
    @pytest.mark.slow
    def test_timelapse_full_analysis(self, structure_timelapse_filepath):
        """Test complete structural analysis pipeline on time-lapse."""
        sarc = Structure(structure_timelapse_filepath, restart=False)
        sarc.detect_sarcomeres(max_patch_size=(1024, 1024))
        sarc.full_analysis_structure()
        
        # Verify analysis results
        assert 'sarcomere_length_vectors' in sarc.data
        assert 'myof_length' in sarc.data
        assert 'domains' in sarc.data


class TestStructureSingleImageAnalysis:
    """Test analysis pipeline on single images."""

    def test_single_image_sarcomere_detection(self, structure_single_filepath):
        """Test sarcomere detection on single image."""
        sarc = Structure(structure_single_filepath, restart=False)
        sarc.detect_sarcomeres(max_patch_size=(1024, 1024))
        
        # Verify detection attributes exist
        assert hasattr(sarc, 'zbands')
        assert hasattr(sarc, 'mbands')
        assert hasattr(sarc, 'cell_mask')
        
    def test_single_image_full_analysis(self, structure_single_filepath):
        """Test complete structural analysis pipeline on single image."""
        sarc = Structure(structure_single_filepath, restart=False)
        sarc.detect_sarcomeres(max_patch_size=(1024, 1024))
        sarc.full_analysis_structure()
        
        # Verify analysis results
        assert 'sarcomere_length_vectors' in sarc.data
        assert 'myof_length' in sarc.data
        assert 'domains' in sarc.data


class TestStructureErrors:
    """Test error handling."""
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            Structure('nonexistent_file.tif')


class TestStructureIntegration:
    """Integration tests combining multiple features."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_complete_workflow_timelapse(self, structure_timelapse_filepath):
        """Test complete Structure workflow on time-lapse."""
        # Initialize with metadata
        sarc = Structure(structure_timelapse_filepath, 
                        experiment_type='timelapse',
                        restart=True)
        
        # Run detection
        sarc.detect_sarcomeres(max_patch_size=(1024, 1024))
        
        # Run full analysis
        sarc.full_analysis_structure()
        
        # Verify all components completed successfully
        assert hasattr(sarc, 'zbands')
        assert 'sarcomere_length_vectors' in sarc.data
        assert 'myof_length' in sarc.data
        assert 'domains' in sarc.data
        assert sarc.metadata.user_info['experiment_type'] == 'timelapse'
        
    @pytest.mark.integration
    def test_complete_workflow_single_image(self, structure_single_filepath):
        """Test complete Structure workflow on single image."""
        # Initialize with metadata
        sarc = Structure(structure_single_filepath, 
                        experiment_type='single_image',
                        restart=True)
        
        # Run detection
        sarc.detect_sarcomeres(max_patch_size=(1024, 1024))
        
        # Run full analysis
        sarc.full_analysis_structure()
        
        # Verify all components completed successfully
        assert hasattr(sarc, 'zbands')
        assert 'sarcomere_length_vectors' in sarc.data
        assert 'myof_length' in sarc.data
        assert 'domains' in sarc.data
        assert sarc.metadata.user_info['experiment_type'] == 'single_image'
