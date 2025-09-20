FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglx-mesa0 libglib2.0-0 libsm6 \
    libxext6 libxrender-dev libgomp1 libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-trained model (build locally first) - Follow README.md
COPY coin_model_final.pt .
COPY training_images_list.json .

COPY app/ ./app/

RUN mkdir -p api_uploads logs

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]