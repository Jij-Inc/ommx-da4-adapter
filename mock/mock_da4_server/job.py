import asyncio
import datetime
from typing import Literal

from .models import QuboRequest, QuboResponse
from .random_qubo_response import random_qubo_response


class Job:
    def __init__(
        self,
        job_id: str,
        qubo_request: QuboRequest,
        start_time: str,
    ):
        self._job_id = job_id
        self._qubo_request = qubo_request
        self._start_time = start_time

    @property
    def status(self) -> Literal["Waiting", "Running", "Done", "Canceled", "Error"]:
        raise NotImplementedError()

    def get_job_status_info(self) -> dict[str, str]:
        return {
            "job_id": self._job_id,
            "job_status": self.status,
            "start_time": self._start_time,
        }


class WaitingJob(Job):
    @property
    def status(self) -> Literal["Waiting"]:
        return "Waiting"

    def run(self) -> "RunningJob":
        return RunningJob(
            job_id=self._job_id,
            qubo_request=self._qubo_request,
            start_time=self._start_time,
        )

    def cancel(self) -> "CanceledJob":
        return CanceledJob(
            job_id=self._job_id,
            qubo_request=self._qubo_request,
            start_time=self._start_time,
        )


class RunningJob(Job):
    @property
    def status(self) -> Literal["Running"]:
        return "Running"

    def done(self) -> "DoneJob":
        return DoneJob(
            job_id=self._job_id,
            qubo_request=self._qubo_request,
            start_time=self._start_time,
        )

    def error(self) -> "ErrorJob":
        return ErrorJob(
            job_id=self._job_id,
            qubo_request=self._qubo_request,
            start_time=self._start_time,
        )


class DoneJob(Job):
    @property
    def status(self) -> Literal["Done"]:
        return "Done"

    def get_result(self) -> QuboResponse:
        return random_qubo_response(qubo_request=self._qubo_request)


class CanceledJob(Job):
    @property
    def status(self) -> Literal["Canceled"]:
        return "Canceled"


class ErrorJob(Job):
    @property
    def status(self) -> Literal["Error"]:
        return "Error"


async def job_runner(
    job_id: str,
    qubo_request: QuboRequest,
    job_store: dict[str, Job],
) -> None:
    WAITING_TIME = 5
    RUNNING_TIME = 10

    waiting_job = WaitingJob(
        job_id=job_id,
        qubo_request=qubo_request,
        start_time=datetime.datetime.now().isoformat(),
    )
    job_store[job_id] = waiting_job
    print(f"[CREATE] job_id: {job_id}, status: {waiting_job.status}")
    await asyncio.sleep(WAITING_TIME)

    if isinstance(job_store[job_id], CanceledJob):
        return

    running_job = waiting_job.run()
    job_store[job_id] = running_job
    print(f"[UPDATE] job_id: {job_id}, status: {running_job.status}")
    await asyncio.sleep(RUNNING_TIME)

    if isinstance(job_store[job_id], CanceledJob):
        return

    done_job = running_job.done()
    job_store[job_id] = done_job
    print(f"[UPDATE] job_id: {job_id}, status: {done_job.status}")
