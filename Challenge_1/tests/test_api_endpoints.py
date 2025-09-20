# challenge1/tests/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io
from PIL import Image
import numpy as np

from app.main import app


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

    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_upload_image_success(self, client, sample_image_file):
        """Test successful image upload"""
        with patch("app.api.endpoints.detector.detect") as mock_detect:
            mock_detect.return_value = []

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

    def test_get_image_coins(self, client):
        """Test retrieving coins for an image"""
        with patch("app.api.endpoints.db") as mock_db:
            mock_query = Mock()
            mock_db.query.return_value.filter_by.return_value.all.return_value = []

            response = client.get("/api/v1/images/test-id/coins")

            assert response.status_code == 404  # No coins found

    def test_evaluate_detection(self, client):
        """Test evaluation endpoint"""
        payload = {
            "predictions": [{"bbox": [10, 10, 50, 50]}],
            "ground_truth": [{"bbox": [12, 12, 48, 48]}],
        }

        response = client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert 0 <= data["precision"] <= 1
        assert 0 <= data["recall"] <= 1
