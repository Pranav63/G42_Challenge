# challenge2/tests/test_image_processor.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from app.services.image_processor import ImageProcessor

class TestImageProcessor:
    
    @pytest.fixture
    def processor(self):
        return ImageProcessor()
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file"""
        data = {
            'depth': np.arange(10),
            **{f'col{i}': np.random.randint(0, 255, 10) for i in range(1, 201)}
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            return f.name
    
    def test_resize_row(self, processor):
        """Test image row resizing from 200 to 150"""
        row = np.random.rand(200)
        resized = processor.resize_image_row(row)
        
        assert len(resized) == 150
        assert resized.min() >= 0
        assert resized.max() <= 255
    
    def test_resize_maintains_pattern(self, processor):
        """Test that resizing maintains general pattern"""
        # Create a gradient pattern
        row = np.linspace(0, 255, 200)
        resized = processor.resize_image_row(row)
        
        assert len(resized) == 150
        # Should still be generally increasing
        assert resized[0] < resized[-1]
    
    def test_process_csv_file(self, processor, sample_csv):
        """Test CSV file processing"""
        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = None
        mock_db.add_all = Mock()
        mock_db.commit = Mock()
        
        result = processor.process_csv_file(sample_csv, mock_db)
        
        assert result['rows_processed'] == 10
        assert 'processing_time_ms' in result
        assert isinstance(result['errors'], list)
        
        # Cleanup
        os.unlink(sample_csv)
    
    def test_apply_colormap(self, processor):
        """Test colormap application"""
        grayscale = np.random.randint(0, 255, (1, 150), dtype=np.uint8)
        
        colored = processor.apply_colormap(grayscale, 'viridis')
        
        assert colored.shape == (1, 150, 3)  # RGB output
        assert colored.dtype == np.uint8
    
    def test_invalid_colormap_defaults_to_viridis(self, processor):
        """Test invalid colormap name defaults to viridis"""
        grayscale = np.random.randint(0, 255, (1, 150), dtype=np.uint8)
        
        colored = processor.apply_colormap(grayscale, 'invalid_colormap')
        
        assert colored.shape == (1, 150, 3)
    
    def test_get_frames_in_range(self, processor):
        """Test retrieving frames within depth range"""
        mock_db = Mock()
        mock_frame = Mock()
        mock_frame.depth = 5.0
        mock_frame.width = 150
        mock_frame.height = 1
        mock_frame.to_array.return_value = np.array([[128] * 150])
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_frame]
        
        frames = processor.get_frames_in_range(mock_db, 0, 10)
        
        assert len(frames) == 1
        assert frames[0]['depth'] == 5.0
        assert frames[0]['width'] == 150
    
    def test_handle_nan_values(self, processor):
        """Test handling of NaN values in data"""
        row_with_nan = np.array([1, 2, np.nan, 4, 5] * 40)  # 200 values
        result = processor.resize_image_row(row_with_nan)
        
        assert not np.isnan(result).any()
        assert len(result) == 150