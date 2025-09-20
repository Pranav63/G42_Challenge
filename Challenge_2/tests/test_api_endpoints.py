# challenge2/tests/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io
import pandas as pd

from app.main import app


class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing"""
        data = {
            "depth": [1.0, 2.0, 3.0],
            **{f"col{i}": [100, 150, 200] for i in range(1, 201)},
        }
        df = pd.DataFrame(data)

        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        return csv_bytes

    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_upload_csv_success(self, client, sample_csv_file):
        """Test successful CSV upload"""
        with patch("app.api.endpoints.processor.process_csv_file") as mock_process:
            mock_process.return_value = {
                "rows_processed": 3,
                "processing_time_ms": 100.0,
                "errors": [],
            }

            response = client.post(
                "/api/v1/upload-csv",
                files={"file": ("test.csv", sample_csv_file, "text/csv")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["rows_processed"] == 3

    def test_upload_non_csv_file(self, client):
        """Test uploading non-CSV file"""
        response = client.post(
            "/api/v1/upload-csv",
            files={"file": ("test.txt", b"text content", "text/plain")},
        )

        assert response.status_code == 400
        assert "must be CSV" in response.json()["detail"]

    def test_get_frames(self, client):
        """Test getting frames by depth range"""
        with patch("app.api.endpoints.processor.get_frames_in_range") as mock_get:
            mock_get.return_value = [
                {"depth": 5.0, "width": 150, "height": 1, "data": [[128] * 150]}
            ]

            response = client.get("/api/v1/frames?depth_min=0&depth_max=10")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["depth"] == 5.0

    def test_get_frames_invalid_range(self, client):
        """Test invalid depth range"""
        response = client.get("/api/v1/frames?depth_min=10&depth_max=5")

        assert response.status_code == 400
        assert "depth_min must be <= depth_max" in response.json()["detail"]

    def test_get_statistics(self, client):
        """Test statistics endpoint"""
        with patch("app.api.endpoints.processor.get_statistics") as mock_stats:
            mock_stats.return_value = {
                "total_frames": 100,
                "depth_range": (0, 100),
                "storage_size_mb": 0.5,
            }

            response = client.get("/api/v1/frames/statistics")

            assert response.status_code == 200
            data = response.json()
            assert data["total_frames"] == 100

    def test_get_colormaps(self, client):
        """Test available colormaps endpoint"""
        response = client.get("/api/v1/colormaps")

        assert response.status_code == 200
        data = response.json()
        assert "colormaps" in data
        assert "viridis" in data["colormaps"]
