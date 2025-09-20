

# Challenge 2: Image Processing Pipeline

A high-performance API for processing CSV-based image data with resizing and colormap application capabilities.

## 🎯 Overview

This solution processes large CSV files containing image data (5461 rows × 200 columns), resizes each row from 200 to 150 pixels using linear interpolation, stores the data efficiently in SQLite, and provides fast retrieval with optional colormap application.

## 🏗️ Architecture

```mermaid
graph LR
    subgraph "Input"
        CSV[CSV File<br/>5461×200]
    end
    
    subgraph "Processing Pipeline"
        P1[Parse CSV]
        P2[Resize Images<br/>200→150]
        P3[Batch Processing]
        P4[Apply Colormap]
    end
    
    subgraph "Storage"
        DB[(SQLite<br/>Binary Storage)]
    end
    
    subgraph "API"
        A1[Upload Endpoint]
        A2[Query Endpoint]
        A3[Statistics]
    end
    
    CSV --> A1
    A1 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> DB
    DB --> A2
    A2 --> P4
    
    style CSV fill:#e1f5fe
    style P2 fill:#fff3e0
    style DB fill:#e8f5e9
```

### Data Flow

```mermaid
flowchart TD
    Start([CSV Upload]) --> Parse[Parse CSV Data]
    Parse --> Validate{Valid Format?}
    Validate -->|No| Error[Return Error]
    Validate -->|Yes| Extract[Extract Pixel Values]
    Extract --> Normalize[Normalize 0-255]
    Normalize --> Resize[Resize 200→150]
    Resize --> Batch[Batch Processing<br/>100 rows/batch]
    Batch --> Store[(Store in SQLite)]
    Store --> Index[Index by Depth]
    Index --> Complete([Ready for Queries])
    
    Query([Query Request]) --> Range[Depth Range Filter]
    Range --> Retrieve[Retrieve from DB]
    Retrieve --> Color{Apply Colormap?}
    Color -->|Yes| Map[Apply Color Mapping]
    Color -->|No| Return1[Return Grayscale]
    Map --> Return2[Return RGB]
```

## 🚀 Quick Start

### Using Docker

```bash
# Build and run
docker-compose up --build

# API will be available at http://localhost:8001
# Swagger docs at http://localhost:8001/docs
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Run application
uvicorn app.main:app --reload --port 8001
```

## 📡 API Endpoints

1. **Upload CSV**  
   - Endpoint: `POST /api/v1/upload-csv`  
   - Content-Type: `multipart/form-data`  
   - Processing: Resizes and stores image data  

2. **Get Frames by Depth**  
   - Endpoint: `GET /api/v1/frames`  
   - Parameters:  
     - `depth_min`: Minimum depth value  
     - `depth_max`: Maximum depth value  
     - `colormap`: Optional (viridis, jet, hot, etc.)  

3. **Get Statistics**  
   - Endpoint: `GET /api/v1/frames/statistics`  
   - Response: Total frames, depth range, storage size  

4. **Available Colormaps**  
   - Endpoint: `GET /api/v1/colormaps`  
   - Response: List of available colormaps  

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v --cov=app --cov-report=html
```

## 📊 Performance Metrics

- Processing Speed: 10–20 rows/ms
- Total Storage: ~800KB for 5461 frames
- Query Speed: < 50ms for range queries
- Batch Size: 100 rows (optimal)

## 🎨 Available Colormaps

- viridis, jet, hot, cool, hsv
- rainbow, plasma, inferno, magma, twilight

## 🛠️ Technology Stack

- FastAPI 0.104.1
- Pandas 2.1.3, NumPy 1.24.3
- OpenCV 4.8.1
- SQLite
- Uvicorn

## 📁 Project Structure

```
challenge2/
├── app/
│   ├── api/          # API endpoints
│   ├── core/         # Configuration
│   ├── models/       # Database models
│   ├── services/     # Processing logic
│   └── main.py       # Application entry
├── tests/            # Unit tests
├── requirements.txt  # Dependencies
└── docker-compose.yml
```
