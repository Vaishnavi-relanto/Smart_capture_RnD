from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
from datetime import datetime
import os
import main
from main import *
# Import your existing classes
from typing import Optional, Dict, Union, List

# Note: Import your existing classes here

app = FastAPI(
    title="Document Processing API",
    description="API for processing documents and extracting events",
    version="1.0.0"
)


class URLInput(BaseModel):
    url: HttpUrl
    output_dir: Optional[str] = "output_pdfs"


class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    url: str
    submitted_at: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None


# Store job statuses in memory (replace with database in production)
jobs = {}


def process_url_background(url: str, output_dir: str, job_id: str):
    try:
        processor = TimedEnhancedDocumentProcessor(debug_mode=True)
        result = processor.process_single_url(url, output_dir)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/process-url", response_model=ProcessingResponse)
async def process_url(url_input: URLInput, background_tasks: BackgroundTasks):
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

    jobs[job_id] = {
        "status": "processing",
        "url": str(url_input.url),
        "submitted_at": datetime.now().isoformat()
    }

    background_tasks.add_task(
        process_url_background,
        str(url_input.url),
        url_input.output_dir,
        job_id
    )

    return ProcessingResponse(
        job_id=job_id,
        status="processing",
        url=str(url_input.url),
        submitted_at=jobs[job_id]["submitted_at"]
    )


@app.get("/job-status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=jobs[job_id]["status"],
        result=jobs[job_id].get("result"),
        error=jobs[job_id].get("error")
    )


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    return [
        JobStatus(
            job_id=job_id,
            status=job_info["status"],
            result=job_info.get("result"),
            error=job_info.get("error")
        )
        for job_id, job_info in jobs.items()
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)