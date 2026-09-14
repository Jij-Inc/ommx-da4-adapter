import uuid
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, status
from fastapi.responses import JSONResponse

from .check import check_accept, check_auth, check_content_type
from .job import DoneJob, Job, WaitingJob, job_runner
from .models import JobID, JobStatus, JobStatusList, QuboRequest

app = FastAPI()
job_store: dict[str, Job] = {}

AuthDependency = Annotated[JSONResponse | None, Depends(check_auth)]
AcceptDependency = Annotated[JSONResponse | None, Depends(check_accept)]
ContentTypeDependency = Annotated[
    JSONResponse | None,
    Depends(check_content_type),
]


@app.get("/v1/healthcheck", tags=["v1"])
def get_v1_healthcheck(
    auth: AuthDependency,
    accept: AcceptDependency,
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
    auth: AuthDependency,
    accept: AcceptDependency,
    content_type: ContentTypeDependency,
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
    auth: AuthDependency,
    accept: AcceptDependency,
) -> JSONResponse:
    if auth is not None:
        return auth
    if accept is not None:
        return accept

    job_status_list = [value.get_job_status_info() for value in job_store.values()]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=JobStatusList(job_status_list=job_status_list).model_dump(),
    )


@app.get("/v4/async/jobs/result/{job_id}", tags=["v4"])
def get_v4_async_jobs_result(
    job_id: str,
    auth: AuthDependency,
    accept: AcceptDependency,
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
    auth: AuthDependency,
    accept: AcceptDependency,
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
    auth: AuthDependency,
    accept: AcceptDependency,
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
