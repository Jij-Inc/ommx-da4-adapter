import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, status
from fastapi.responses import JSONResponse

from .check import check_auth, check_accept, check_content_type
from .job import Job, job_runner, DoneJob, WaitingJob
from .models import QuboRequest, JobStatusList, JobID, JobStatus


app = FastAPI()
job_store: dict[str, Job] = {}


@app.get("/v1/healthcheck", tags=["v1"])
def get_v1_healthcheck(
    auth: JSONResponse | None = Depends(check_auth),
    accept: JSONResponse | None = Depends(check_accept),
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={},
    )


@app.post("/v4/async/qubo/solve", tags=["v4"])
def post_v4_async_qubo_solve(
    qubo_request: QuboRequest,
    background_tasks: BackgroundTasks,
    auth: JSONResponse | None = Depends(check_auth),
    accept: JSONResponse | None = Depends(check_accept),
    content_type: JSONResponse | None = Depends(check_content_type),
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept
    if content_type is not None:
        return content_type

    # TODO: Write a process to return an error when the number of jobs exceeds a certain limit

    job_id = uuid.uuid4().hex
    background_tasks.add_task(job_runner, job_id, qubo_request, job_store)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=JobID(job_id=job_id).model_dump(),
    )


@app.get("/v4/async/jobs", tags=["v4"])
def get_v4_async_jobs(
    auth: JSONResponse | None = Depends(check_auth),
    accept: JSONResponse | None = Depends(check_accept),
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept

    job_status_list = []
    for _, value in job_store.items():
        job_status_list.append(value.get_job_status_info())

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=JobStatusList(job_status_list=job_status_list).model_dump(),
    )


@app.get("/v4/async/jobs/result/{job_id}", tags=["v4"])
def get_v4_async_jobs_result(
    job_id: str,
    auth: JSONResponse | None = Depends(check_auth),
    accept: JSONResponse | None = Depends(check_accept),
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept

    if job_id not in job_store:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": 404,
                    "title": "Resource Not Found",
                    "message": "Resource not found.",
                }
            },
        )

    job = job_store[job_id]
    if isinstance(job, DoneJob):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=job.get_result().model_dump(),
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=JobStatus(status=job.status).model_dump(),
        )


@app.delete("/v4/async/jobs/result/{job_id}", tags=["v4"])
def delete_v4_async_jobs_result(
    job_id: str,
    auth: JSONResponse | None = Depends(check_auth),
    accept: JSONResponse | None = Depends(check_accept),
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept

    if job_id not in job_store:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": 404,
                    "title": "Resource Not Found",
                    "message": "Resource not found.",
                }
            },
        )

    job = job_store.pop(job_id)
    if isinstance(job, DoneJob):
        qubo_response = job.get_result()
        qubo_response.status = "Deleted"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=qubo_response.model_dump(),
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=JobStatus(status=job.status).model_dump(),
        )


@app.post("/v4/async/jobs/cancel", tags=["v4"])
def post_v4_async_jobs_cancel(
    cancel_request: JobID,
    auth: JSONResponse | None = Depends(check_auth),
    accept: JSONResponse | None = Depends(check_accept),
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept

    job_id = cancel_request.job_id
    if job_id not in job_store:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": {
                    "code": 404,
                    "title": "Resource Not Found",
                    "message": "Resource not found.",
                }
            },
        )

    job = job_store[job_id]
    if isinstance(job, WaitingJob):
        job_store[job_id] = job.cancel()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=JobStatus(status=job_store[job_id].status).model_dump(),
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=JobStatus(status=job.status).model_dump(),
        )
