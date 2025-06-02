from flask import Flask, request, jsonify
from transformers import pipeline
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import queue

app = Flask(__name__)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

jobs = {}
job_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=4)

CLEANUP_INTERVAL = 300 
RESULT_TTL = 3600 

def cleanup_jobs():
    while True:
        now = time.time()
        to_delete = []
        for job_id, job in list(jobs.items()):
            if job.get('status') == 'completed' and 'completed_at' in job:
                if now - job['completed_at'] > RESULT_TTL:
                    to_delete.append(job_id)
            if job.get('status') == 'failed' and 'completed_at' in job:
                if now - job['completed_at'] > RESULT_TTL:
                    to_delete.append(job_id)
        for job_id in to_delete:
            del jobs[job_id]
        time.sleep(CLEANUP_INTERVAL)

cleanup_thread = threading.Thread(target=cleanup_jobs, daemon=True)
cleanup_thread.start()

def process_classification_job(job_id, titles, categories):
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['started_at'] = time.time()
    
    try:
        results = []
        for i, title in enumerate(titles):
            result = classifier(title, categories)
            results.append({
                "title": title,
                "predicted": result["labels"][0],
                "scores": dict(zip(result["labels"], result["scores"]))
            })
            jobs[job_id]['progress'] = (i + 1) / len(titles) * 100
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['results'] = results
        jobs[job_id]['completed_at'] = time.time()
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

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
    while True:
        if job_id not in jobs:
            break
        job_id = str(uuid.uuid4())


    jobs[job_id] = {
        'status': 'queued',
        'created_at': time.time(),
        'progress': 0,
        'total': len(titles),
        'categories': categories
    }
    
    executor.submit(process_classification_job, job_id, titles, categories)
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'total': len(titles)
    }), 202

@app.route('/classify', methods=['POST'])
def classify_single():
    data = request.json
    title = data.get('title')
    categories = data.get('categories', [])
    
    if not title or not categories:
        return jsonify({'error': 'Title and categories required'}), 400
    
    result = classifier(title, categories)
    
    return jsonify({
        "title": title,
        "categories": categories,
        "predicted": result["labels"][0],
        "scores": dict(zip(result["labels"], result["scores"]))
    })

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id].copy()
    
    # Add timing info
    if 'started_at' in job and 'completed_at' not in job:
        job['duration'] = time.time() - job['started_at']
    elif 'completed_at' in job:
        job['duration'] = job['completed_at'] - job['started_at']
    
    return jsonify(job)

@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    return jsonify({
        'job_id': job_id,
        'results': job['results'],
        'total': job['total'],
        'categories': job['categories']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'active_jobs': len([j for j in jobs.values() if j['status'] in ['queued', 'processing']]),
        'total_jobs': len(jobs)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)