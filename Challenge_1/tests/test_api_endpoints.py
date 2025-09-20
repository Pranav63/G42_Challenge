import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import io
from PIL import Image
import numpy as np
import uuid

from app.main import app
from app.models import database


class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing"""
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        return MagicMock()

    @pytest.fixture(autouse=True)
    def override_get_db(self, mock_db_session):
        """Override the get_db dependency for all tests"""
        def mock_get_db():
            return mock_db_session
        
        app.dependency_overrides[database.get_db] = mock_get_db
        yield
        app.dependency_overrides.clear()

    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch('app.api.endpoints.storage')
    def test_upload_image_success(self, mock_storage, client, sample_image_file, mock_db_session):
        """Test successful image upload"""
        # Generate unique ID for this test
        unique_id = str(uuid.uuid4())
        
        # Mock storage service
        mock_storage.save_image.return_value = (unique_id, f"/path/to/{unique_id}.jpg")
        mock_storage.bytes_to_array.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock the YOLO detector instead of fallback detector
        with patch('app.api.endpoints.yolo_detector') as mock_yolo:
            mock_yolo.detect.return_value = ([], False)
            
            response = client.post(
                "/api/v1/images/upload",
                files={"file": ("test.jpg", sample_image_file, "image/jpeg")},
            )

            assert response.status_code == 200
            data = response.json()
            assert "image_id" in data
            assert "coin_count" in data
            assert data["coin_count"] == 0

    def test_upload_invalid_file_type(self, client):
        """Test uploading non-image file"""
        response = client.post(
            "/api/v1/images/upload",
            files={"file": ("test.txt", b"text content", "text/plain")},
        )

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    def test_get_image_coins(self, client, mock_db_session):
        """Test retrieving coins for an image"""
        # Mock query to return empty list (no coins found)
        mock_db_session.query.return_value.filter_by.return_value.all.return_value = []

        response = client.get("/api/v1/images/test-id/coins")

        assert response.status_code == 404  # No coins found

    def test_get_image_coins_success(self, client, mock_db_session):
        """Test retrieving coins for an image - success case"""
        # Mock coin data
        mock_coin = MagicMock()
        mock_coin.id = "coin-1"
        mock_coin.bbox_x = 10
        mock_coin.bbox_y = 10
        mock_coin.bbox_width = 50
        mock_coin.bbox_height = 50
        mock_coin.centroid_x = 35
        mock_coin.centroid_y = 35
        mock_coin.radius = 25
        mock_coin.confidence = 0.9
        
        mock_db_session.query.return_value.filter_by.return_value.all.return_value = [mock_coin]

        response = client.get("/api/v1/images/test-id/coins")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "coin-1"

    def test_evaluate_detection(self, client):
        """Test evaluation endpoint"""
        payload = {
            "predictions": [{"bbox": [10, 10, 50, 50]}],
            "ground_truth": [{"bbox": [12, 12, 48, 48]}],
            "iou_threshold": 0.5
        }

        with patch('app.api.endpoints.Evaluator') as mock_evaluator_class:
            # Mock the evaluator instance and its evaluate method
            mock_evaluator = MagicMock()
            mock_evaluator_class.return_value = mock_evaluator

            mock_evaluator.evaluate.return_value = {
                "precision": 0.8,
                "recall": 0.9,
                "f1_score": 0.85,
                "true_positives": 4,
                "false_positives": 1,
                "false_negatives": 1
            }

            response = client.post("/api/v1/evaluate", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert "precision" in data
            assert "recall" in data
            assert "f1_score" in data
            assert "true_positives" in data
            assert "false_positives" in data
            assert "false_negatives" in data
            assert 0 <= data["precision"] <= 1
            assert 0 <= data["recall"] <= 1