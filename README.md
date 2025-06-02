# Titles Classification Service

A Flask-based API for zero-shot classification of text titles using Hugging Face Transformers (`facebook/bart-large-mnli`).  
Supports single and batch classification with job management and health endpoints.

[Docker Image](https://hub.docker.com/r/nazaua/bart-classification-service)

## Features

- **Zero-shot classification** of titles into user-provided categories
- **Batch processing** with job status tracking
- **Job results retrieval**
- **Health check endpoint**
- **Dockerized** for easy deployment

---

## API Endpoints

### 1. Classify Single Title

**POST** `/classify`

**Request Body:**
```json
{
  "title": "Your title here",
  "categories": ["Category1", "Category2", "Category3"]
}
```

**Response:**
```json
{
  "title": "Your title here",
  "categories": ["Category1", "Category2", "Category3"],
  "predicted": "Category1",
  "scores": {
    "Category1": 0.95,
    "Category2": 0.03,
    "Category3": 0.02
  }
}
```

---

### 2. Classify Batch of Titles

**POST** `/classify/batch`

**Request Body:**
```json
{
  "titles": ["Title 1", "Title 2", "Title 3"],
  "categories": ["Category1", "Category2", "Category3"]
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "total": 3
}
```

---

### 3. Get Job Status

**GET** `/jobs/<job_id>`

**Response:**
```json
{
  "status": "processing",
  "created_at": 1717320000.0,
  "progress": 66.7,
  "total": 3,
  "categories": ["Category1", "Category2", "Category3"],
  "started_at": 1717320001.0,
  "duration": 2.5
}
```

---

### 4. Get Job Results

**GET** `/jobs/<job_id>/results`

**Response:**
```json
{
  "job_id": "uuid-string",
  "results": [
    {
      "title": "Title 1",
      "predicted": "Category1",
      "scores": {
        "Category1": 0.95,
        "Category2": 0.03,
        "Category3": 0.02
      }
    },
    ...
  ],
  "total": 3,
  "categories": ["Category1", "Category2", "Category3"]
}
```

---

### 5. Health Check

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "active_jobs": 1,
  "total_jobs": 5
}
```

---

## Running with Docker

```bash
docker build -t titles-classification .
docker run -p 8000:8000 -e NUMBER_GUNICORN_WORKERS=1 titles-classification:latest
```

### NUMBER_GUNICORN_WORKERS

If you deploy this service with [Gunicorn](https://gunicorn.org/), you can control the number of worker processes using the `NUMBER_GUNICORN_WORKERS` environment variable.

- **Purpose:**  
  Limits Gunicorn to a single worker process, which is recommended for memory-intensive ML models. This ensures the model is loaded only once and avoids excessive memory usage.

- **Usage with Gunicorn:**  
  If you want to run the service with Gunicorn instead of Flask’s built-in server, use:
  ```bash
  gunicorn -w $NUMBER_GUNICORN_WORKERS -b 0.0.0.0:8000 ml_service:app
  ```

## Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)

## Notes

- Maximum 100 titles per batch request.
- Results are kept for 1 hour after completion.
- The model is loaded once and shared across all requests for efficiency.

---

## License

MIT License