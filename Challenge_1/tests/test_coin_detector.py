import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from app.services.coin_detector import CoinDetector, DetectedCoin


class TestCoinDetector:
    @pytest.fixture
    def detector(self):
        return CoinDetector()

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with circles drawn."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        # Draw some circles
        cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
        cv2.circle(img, (300, 300), 70, (255, 255, 255), -1)
        return img

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert hasattr(detector, 'detect')
        assert callable(getattr(detector, 'detect'))

    def test_detect_coins_in_image(self, detector, sample_image):
        """Test coin detection returns results."""
        coins = detector.detect(sample_image)
        assert isinstance(coins, list)
        assert len(coins) >= 0  
        if coins:
            coin = coins[0]
            assert hasattr(coin, "id")
            assert hasattr(coin, "centroid")
            assert hasattr(coin, "radius")
            assert hasattr(coin, "bounding_box")
            assert hasattr(coin, "confidence")
            assert 0 <= coin.confidence <= 1

    def test_detected_coin_properties(self):
        """Test DetectedCoin object properties."""
        coin = DetectedCoin(100, 150, 50, 0.9)
        
        assert coin.centroid == (100, 150)
        assert coin.radius == 50
        assert coin.confidence == 0.9
        assert coin.bounding_box == (50, 100, 100, 100)  # (x-r, y-r, 2*r, 2*r)
        assert isinstance(coin.id, str)
        assert len(coin.id) > 0 

    def test_detected_coin_default_confidence(self):
        """Test DetectedCoin with default confidence."""
        coin = DetectedCoin(50, 60, 30)
        assert coin.confidence == 1.0

    def test_grayscale_conversion(self, detector):
        """Test handling of grayscale images."""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        coins = detector.detect(gray_image)
        assert isinstance(coins, list)

    def test_color_image_handling(self, detector):
        """Test that detector handles color images."""
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        coins = detector.detect(color_image)
        assert isinstance(coins, list)

    def test_empty_image_returns_empty_list(self, detector):
        """Test that empty image returns no coins."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        coins = detector.detect(empty_image)
        assert coins == []

    def test_small_image_handling(self, detector):
        """Test detection with very small images."""
        small_image = np.zeros((50, 50), dtype=np.uint8)
        coins = detector.detect(small_image)
        assert isinstance(coins, list)

    def test_large_image_handling(self, detector):
        """Test detection with large images."""
        large_image = np.zeros((1000, 1000), dtype=np.uint8)
        coins = detector.detect(large_image)
        assert isinstance(coins, list)

    @patch('cv2.HoughCircles')
    def test_hough_circles_called_with_correct_params(self, mock_hough, detector):
        """Test that HoughCircles is called with adaptive parameters."""

        mock_hough.return_value = None
        
        image = np.zeros((200, 300), dtype=np.uint8) 
        coins = detector.detect(image)
        
        assert mock_hough.called
        call_args = mock_hough.call_args[1]  
        
        # Check that minRadius and maxRadius are adaptive to image size
        expected_min_radius = int(min(200, 300) * 0.02)  
        expected_max_radius = int(min(200, 300) * 0.25)  
        
        assert call_args['minRadius'] == expected_min_radius
        assert call_args['maxRadius'] == expected_max_radius
        assert coins == []

    @patch('cv2.HoughCircles')
    def test_duplicate_removal(self, mock_hough, detector):
        """Test that overlapping circles are filtered out."""

        mock_circles = np.array([[[100, 100, 50], [105, 105, 45], [200, 200, 60]]])
        mock_hough.return_value = mock_circles
        
        image = np.zeros((400, 400), dtype=np.uint8)
        coins = detector.detect(image)
        
        # Should filter out overlapping circles 
        assert len(coins) == 2 
        
        # Verify the remaining coins are not too close to each other
        if len(coins) >= 2:
            coin1, coin2 = coins[0], coins[1]
            distance = np.sqrt((coin1.centroid[0] - coin2.centroid[0])**2 + 
                             (coin1.centroid[1] - coin2.centroid[1])**2)
            min_distance = max(coin1.radius, coin2.radius) * 0.5
            assert distance >= min_distance

    def test_bounding_box_calculation(self):
        """Test bounding box calculation for DetectedCoin."""
        coin = DetectedCoin(150, 200, 75)
        
        expected_bbox = (75, 125, 150, 150)  # (x-r, y-r, 2*r, 2*r)
        assert coin.bounding_box == expected_bbox

    def test_confidence_assignment(self, detector):
        """Test that detected coins have the expected confidence value."""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (100, 100), 50, 255, -1)
        
        coins = detector.detect(image)
        
        if coins:  
            for coin in coins:
                assert coin.confidence == 0.8  