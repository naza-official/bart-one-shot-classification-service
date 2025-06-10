from flask import Flask, request, jsonify
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from enum import Enum
import threading
import uuid
import time
import logging
import os
import signal
import sys
import atexit
from functools import lru_cache

MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 1))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class JobState(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

@dataclass
class JobStatus:
    status: JobState
    created_at: float
    progress: float
    total: int
    categories: List[str]
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    log: Optional[str] = None

app = Flask(__name__)

jobs: Dict[str, JobStatus] = {}

job_cancel_events: Dict[str, threading.Event] = {}
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
CLEANUP_INTERVAL = 300
RESULT_TTL = 3600
shutdown_event = threading.Event()

@lru_cache(maxsize=1)
def get_classifier():
    import torch
    from transformers import pipeline
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli",
        device=DEVICE
    )
    return classifier

def cleanup_jobs():
    while not shutdown_event.is_set():
        now = time.time()
        to_delete = []
        for job_id, job in list(jobs.items()):
            if job.status in [JobState.COMPLETED, JobState.FAILED, JobState.ABORTED] and job.completed_at:
                if now - job.completed_at > RESULT_TTL:
                    to_delete.append(job_id)
        
        for job_id in to_delete:
            if job_id in jobs:
                del jobs[job_id]
            if job_id in job_cancel_events:
                del job_cancel_events[job_id]
        
        shutdown_event.wait(timeout=CLEANUP_INTERVAL)

def process_classification_job(job_id: str, titles: List[str], categories: List[str]) -> Dict[str, Any]:
    import io

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger(f"worker-{job_id}")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f"Starting classification job {job_id} with {len(titles)} titles")

    classifier = get_classifier()
    
    try:
        results = []
        for i, title in enumerate(titles):
            result = classifier(title, categories)
            results.append({
                "title": title,
                "predicted": result["labels"][0],
                "scores": dict(zip(result["labels"], result["scores"]))
            })
        
        logger.info(f"Job {job_id} completed successfully")
        handler.flush()
        return {
            'status': JobState.COMPLETED.value,
            'results': results,
            'log': log_stream.getvalue()
        }
    except Exception as e:
        logger.exception(f"Job {job_id} failed with error: {str(e)}")
        handler.flush()
        return {
            'status': JobState.FAILED.value,
            'error': str(e),
            'log': log_stream.getvalue()
        }

@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    data = request.json
    titles = data.get('titles', [])
    categories = data.get('categories', [])

    if not titles or not categories:
        return jsonify({'error': 'Titles and categories required'}), 400
    if len(titles) > 100:
        return jsonify({'error': 'Maximum 100 titles allowed'}), 400

    job_id = str(uuid.uuid4())
    while job_id in jobs:
        job_id = str(uuid.uuid4())


    cancel_event = threading.Event()
    job_cancel_events[job_id] = cancel_event

    jobs[job_id] = JobStatus(
        status=JobState.QUEUED,
        created_at=time.time(),
        progress=0,
        total=len(titles),
        categories=categories
    )

    def callback(future: Future):
        try:
            result = future.result()
            if job_id in jobs:  
                jobs[job_id].status = JobState(result['status'])
                jobs[job_id].results = result.get('results')
                jobs[job_id].error = result.get('error')
                jobs[job_id].log = result.get('log')
                jobs[job_id].completed_at = time.time()
        except Exception as e:
            if job_id in jobs:
                jobs[job_id].status = JobState.FAILED
                jobs[job_id].error = str(e)
                jobs[job_id].completed_at = time.time()

    jobs[job_id].status = JobState.PROCESSING
    jobs[job_id].started_at = time.time()
    future = executor.submit(process_classification_job, job_id, titles, categories)
    future.add_done_callback(callback)

    return jsonify({
        'job_id': job_id,
        'status': JobState.PROCESSING.value,
        'total': len(titles)
    }), 202

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    job_dict = asdict(job)
    job_dict['status'] = job.status.value
    
    if job.started_at and not job.completed_at:
        job_dict['duration'] = time.time() - job.started_at
    elif job.completed_at and job.started_at:
        job_dict['duration'] = job.completed_at - job.started_at
    
    return jsonify(job_dict)

@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job.status != JobState.COMPLETED:
        return jsonify({'error': f'Job not completed yet (status: {job.status.value})'}), 400

    return jsonify({
        'job_id': job_id,
        'results': job.results,
        'total': job.total,
        'categories': job.categories
    })

@app.route('/jobs/<job_id>/log', methods=['GET'])
def get_job_log(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({
        'job_id': job_id,
        'log': jobs[job_id].log or ""
    })

@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job.status not in [JobState.QUEUED, JobState.PROCESSING]:
        return jsonify({'error': f'Cannot cancel job in state: {job.status.value}'}), 400
    
    if job_id in job_cancel_events:
        job_cancel_events[job_id].set()
    
    jobs[job_id].status = JobState.ABORTED
    jobs[job_id].completed_at = time.time()
    
    return jsonify({
        'job_id': job_id,
        'status': JobState.ABORTED.value,
        'message': 'Job cancellation requested'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'active_jobs': len([j for j in jobs.values() if j.status in [JobState.QUEUED, JobState.PROCESSING]]),
        'total_jobs': len(jobs)
    })

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    shutdown_gracefully()

def shutdown_gracefully():
    global executor, shutdown_event

    logger.info("Initiating graceful shutdown...")
    shutdown_event.set()

    for job_id, job in jobs.items():
        if job.status in [JobState.QUEUED, JobState.PROCESSING]:
            job.status = JobState.ABORTED
            job.completed_at = time.time()
            if job_id in job_cancel_events:
                job_cancel_events[job_id].set()

    logger.info("Shutting down ProcessPoolExecutor...")
    try:
        executor.shutdown(wait=False, cancel_futures=True)
        
        time.sleep(1)
        
        if hasattr(executor, '_processes') and executor._processes:
            for pid, process in executor._processes.items():
                if process.is_alive():
                    logger.info(f"Terminating process {pid}")
                    process.terminate()
            
            time.sleep(0.5)
            
            for pid, process in executor._processes.items():
                if process.is_alive():
                    logger.info(f"Killing process {pid}")
                    process.kill()
    
    except Exception as e:
        logger.error(f"Error during executor shutdown: {e}")
    
    logger.info("ProcessPoolExecutor shutdown complete")
    
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

atexit.register(shutdown_gracefully)

if __name__ == '__main__':
    try:
        cleanup_thread = threading.Thread(target=cleanup_jobs, daemon=True)
        cleanup_thread.start()
        logger.info(f"Starting Flask app with {MAX_WORKERS} workers")
        app.run(host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
        shutdown_gracefully()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        shutdown_gracefully()