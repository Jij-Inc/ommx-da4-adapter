import time
from typing import Any, Literal, Optional
import requests

from .exception import OMMXDA4AdapterError
from .models import QuboRequest, QuboResponse, JobID


class DA4Client:
    """A Client class for Fujitsu Digital Annealer API (DA4).

    This class provides an interface for interacting with the Digital Annealer API.
    """

    def __init__(
        self,
        *,
        token: str,
        url: str = "https://api.aispf.global.fujitsu.com/da",
        version: Literal["v4", "v3c"] = "v4",
    ):
        """Initialize DA4 Client.

        :param token: Authentication token for DA4 API.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :raises OMMXDA4AdapterError: If the version is not "v4" or "v3c"
        """
        if version not in set(["v4", "v3c"]):
            raise OMMXDA4AdapterError(
                "The version of Digital Annealer must be either 'v4' or 'v3c'."
            )

        self._token = token
        self._url = url
        self._headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": self._token,
        }
        self._version = version
        self._api_url_jobs = f"/{self._version}/async/jobs"

        # The max trial count to access the Fujitsu Digital Annealer.
        self._max_loop_count = 10000

        # The access interval [sec] to access the Fujitsu Digital Annealer.
        self._solver_access_interval = 10

    def get(self, url: str) -> dict[str, Any]:
        """wrapper of `requests.get`

        :param url: The endpoint path
        :return: json dict
        :raises OMMXDA4AdapterError: If an HTTP status other than the 200s is returned.
        """
        response = requests.get(self._url + url, headers=self._headers)
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if e.response is not None:
                raise OMMXDA4AdapterError(e.response.text) from e
            else:
                raise OMMXDA4AdapterError(str(e)) from e
        return response.json()

    def post(
        self, url: str, serialized_body: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        """wrapper of `requests.post`.

        :param url: The endpoint path
        :param serialized_body: JSON string to send as the request body
        :param headers: HTTP request headers
        :return: json dict
        :raises OMMXDA4AdapterError: If an HTTP status other than 200s is returned
        """
        response = requests.post(self._url + url, headers=headers, data=serialized_body)
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if e.response is not None:
                raise OMMXDA4AdapterError(e.response.text) from e
            else:
                raise OMMXDA4AdapterError(str(e)) from e
        return response.json()

    def delete(self, url: str) -> dict[str, Any]:
        """wrapper of `requests.delete`.

        :param url: The endpoint path
        :return: json dict
        :raises OMMXDA4AdapterError: If an HTTP status other than 200s is returned.
        """
        response = requests.delete(self._url + url, headers=self._headers)
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if e.response is not None:
                raise OMMXDA4AdapterError(e.response.text) from e
            else:
                raise OMMXDA4AdapterError(str(e)) from e
        return response.json()

    def post_qubo_solve(
        self,
        qubo_request: QuboRequest,
        *,
        blob_sas_token: Optional[str] = None,
        blob_account_name: Optional[str] = None,
    ) -> str:
        """Post QUBO problem to the DA4 solver with optional Azure Blob Storage support.

        :param qubo_request: The QUBO problem data
        :param blob_sas_token: X-Blob-Sas-Token. Defaults to None.
        :param blob_account_name: X-Storage-Account-Name. Defaults to None.
        :return: The job ID
        """
        serialized_body = qubo_request.model_dump_json(exclude_none=True, by_alias=True)

        if (blob_sas_token is not None) and (blob_account_name is not None):
            da4_headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "X-Api-Key": self._token,
                "X-Blob-Sas-Token": blob_sas_token,
                "X-Storage-Account-Name": blob_account_name,
            }
        else:
            da4_headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "X-Api-Key": self._token,
            }

        api_url = f"/{self._version}/async/qubo/solve"

        response = self.post(api_url, serialized_body, da4_headers)
        return response["job_id"]

    def get_jobs(self) -> list[dict[str, str]]:
        """Get a list of registered jobs for DA4.

        :return: A list of registered jobs.
        Example response::

            [
                {
                    "job_id": "contract-a001-1234-5678-1a2b3c4d5e6f-123456789012345",
                    "job_status": "Done",
                    "start_time": "2018-07-10T06:02:22Z"
                },
                {
                    "job_id": "contract-a001-1234-5678-1a2b3c4d5e6f-234567890123456",
                    "job_status": "Running",
                    "start_time": "2018-07-10T06:02:24Z"
                },
            ]
        """
        response = self.get(self._api_url_jobs)
        return response["job_status_list"]

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        """Get job results from DA4.

        :param job_id: The job ID
        :return: The job result data
        """
        api_url = self._api_url_jobs + f"/result/{job_id}"
        return self.get(api_url)

    def delete_job_result(self, job_id: str) -> bool:
        """Delete job in DA4.

        :param job_id: The job ID
        :return: True if deletion was successful, False otherwise
        :raises OMMXDA4AdapterError: If an HTTP status other than 200s is returned
        """
        api_url = self._api_url_jobs + f"/result/{job_id}"
        try:
            self.delete(api_url)
            return True
        except OMMXDA4AdapterError:
            return False

    def post_job_cancel(self, job_id: str) -> bool:
        """Cancel job in DA4.

        :param job_id: The job ID
        :return: True if cancellation was successful, False otherwise
        :raises OMMXDA4AdapterError: If an HTTP status other than 200s is returned
        """
        api_url = self._api_url_jobs + "/cancel"
        serialized_body = JobID(job_id=job_id).model_dump_json()
        try:
            self.post(api_url, serialized_body, self._headers)
            return True
        except OMMXDA4AdapterError:
            return False

    def delete_all_jobs(self) -> None:
        """Delete all jobs in DA4."""
        job_list = self.get_jobs()
        for job in job_list:
            self.delete_job_result(job["job_id"])
        return None

    def cancel_all_jobs(self) -> None:
        """Cancel all jobs in DA4."""
        job_list = self.get_jobs()
        for job in job_list:
            self.post_job_cancel(job["job_id"])
        return None

    def fetch_job_result(self, job_id: str) -> QuboResponse:
        """Fetch job result while waiting for completion.

        This method polls the job status until completion and returns the result.
        When completed, it automatically deletes the job.

        :arg job_id: The job id.
        :return: Result returned from DA4.
        :raises OMMXDA4AdapterError: Raised when solving fails for some reason on DA4.

        """
        for _ in range(self._max_loop_count):
            job_result = self.get_job_result(job_id)
            job_status = job_result["status"]

            if job_status == "Done":
                self.delete_job_result(job_id)
                return QuboResponse(**job_result)
            elif job_status == "Error":
                raise OMMXDA4AdapterError(f"Failed to solve: `{job_result}`.")
            elif (job_status == "Running") or (job_status == "Waiting"):
                time.sleep(self._solver_access_interval)
            else:
                raise OMMXDA4AdapterError(
                    f"Unknown error occurred. The status `{job_status}` is unknown."
                )

        delete_success = self.delete_job_result(job_id)
        cancel_success = self.post_job_cancel(job_id)

        if delete_success:
            raise OMMXDA4AdapterError(
                f"Too long solving time. Job id `{job_id}` has been deleted. Sorry."
            )
        elif cancel_success:
            raise OMMXDA4AdapterError(
                f"Too long solving time. Job id `{job_id}` has been canceled. Sorry."
            )
        else:
            raise OMMXDA4AdapterError(
                f"Too long solving time. Job id `{job_id}` is still running."
            )

    def sample(
        self,
        qubo_request: QuboRequest,
        *,
        blob_sas_token: Optional[str] = None,
        blob_account_name: Optional[str] = None,
    ) -> QuboResponse:
        """Sample the result in DA4.

        :param qubo_request: The QUBO problem data
        :param blob_sas_token: X-Blob-Sas-Token. Defaults to None.
        :param blob_account_name: X-Storage-Account-Name. Defaults to None.
        :return: Result returned from DA4.
        """
        posted_job_id = self.post_qubo_solve(
            qubo_request=qubo_request,
            blob_sas_token=blob_sas_token,
            blob_account_name=blob_account_name,
        )

        return self.fetch_job_result(posted_job_id)
