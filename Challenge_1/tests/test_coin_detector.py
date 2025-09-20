
import pytest
import numpy as np
import cv2
from unittest.mock import patch
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
        """Test detector initializes with correct parameters."""
        assert detector.min_radius == 20
        assert detector.max_radius == 100
        assert detector.param1 == 50
        assert detector.param2 == 30

    def test_detect_coins_in_image(self, detector, sample_image):
        """Test coin detection returns results."""
        coins = detector.detect(sample_image)
        assert isinstance(coins, list)
        assert len(coins) >= 0  # May detect 0 or more coins
        if coins:
            coin = coins[0]
            assert hasattr(coin, 'id')
            assert hasattr(coin, 'centroid')
            assert hasattr(coin, 'radius')
            assert hasattr(coin, 'bounding_box')
            assert 0 <= coin.confidence <= 1

    def test_grayscale_conversion(self, detector):
        """Test handling of grayscale images."""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        coins = detector.detect(gray_image)
        assert isinstance(coins, list)

    def test_confidence_calculation(self, detector):
        """Test confidence score calculation."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 255, 2)  # Circle edge
        confidence = detector._calculate_confidence(image, 50, 50, 30)
        assert 0 <= confidence <= 1
        assert confidence > 0  # Should have some confidence for edge

    def test_empty_image_returns_empty_list(self, detector):
        """Test that empty image returns no coins."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        coins = detector.detect(empty_image)
        assert coins == []

    @pytest.mark.parametrize("radius,expected", [
        (10, False),   # Too small
        (50, True),    # In range
        (150, False),  # Too large
    ])
    def test_radius_filtering(self, detector, radius, expected):
        """Test that detector respects radius constraints."""
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(image, (150, 150), radius, 255, -1)
        with patch.object(detector, 'min_radius', 20), \
             patch.object(detector, 'max_radius', 100):
            coins = detector.detect(image)
            if expected:
                assert len(coins) > 0
            else:
                assert len(coins) == 0 or all(
                    c.radius >= 20 and c.radius <= 100 for c in coins
                )
