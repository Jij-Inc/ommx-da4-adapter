import unittest
from unittest.mock import patch, MagicMock, call

import json
import requests

from ommx_da4_adapter.client import DA4Client
from ommx_da4_adapter.exception import OMMXDA4AdapterError
from ommx_da4_adapter.models import (
    BinaryPolynomial,
    BinaryPolynomialTerm,
    FujitsuDA3Solver,
    JobID,
    QuboRequest,
)


class TestDA4Client(unittest.TestCase):
    def setUp(self):
        self.token = "test-token"
        self.url = "https://api.test.com/da"
        self.version = "v4"

        # Mock all HTTP methods of the requests module for the entire test
        patcher = patch("requests.get")
        self.mock_get = patcher.start()
        self.addCleanup(patcher.stop)

        patcher = patch("requests.post")
        self.mock_post = patcher.start()
        self.addCleanup(patcher.stop)

        patcher = patch("requests.delete")
        self.mock_delete = patcher.start()
        self.addCleanup(patcher.stop)

        # Set up default mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        self.mock_get.return_value = mock_response
        self.mock_post.return_value = mock_response
        self.mock_delete.return_value = mock_response

        self.client = DA4Client(token=self.token, url=self.url, version=self.version)
        self.expected_headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": self.token,
        }

    def test_init_valid_version(self):
        client = DA4Client(token=self.token, url=self.url, version="v4")
        self.assertEqual(client._version, "v4")

        client = DA4Client(token=self.token, url=self.url, version="v3c")
        self.assertEqual(client._version, "v3c")

    @patch("requests.get")
    def test_get_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response

        result = self.client.get("/test/endpoint")

        mock_get.assert_called_once_with(
            self.url + "/test/endpoint", headers=self.expected_headers
        )
        self.assertEqual(result, {"test": "data"})

    @patch("requests.get")
    def test_get_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException
        )
        mock_response.text = "Error message"
        mock_get.return_value = mock_response

        with self.assertRaises(OMMXDA4AdapterError):
            self.client.get("/test/endpoint")

    @patch("requests.post")
    def test_post_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_post.return_value = mock_response

        test_data = json.dumps({"key": "value"})
        test_headers = {"header": "value"}

        result = self.client.post("/test/endpoint", test_data, test_headers)

        mock_post.assert_called_once_with(
            self.url + "/test/endpoint",
            headers=test_headers,
            data=test_data,
        )
        self.assertEqual(result, {"test": "data"})

    @patch("requests.post")
    def test_post_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException
        )
        mock_response.text = "Error message"
        mock_post.return_value = mock_response

        test_data = json.dumps({"key": "value"})
        test_headers = {"header": "value"}

        with self.assertRaises(OMMXDA4AdapterError):
            self.client.post("/test/endpoint", test_data, test_headers)

    @patch("requests.delete")
    def test_delete_success(self, mock_delete):
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_delete.return_value = mock_response

        result = self.client.delete("/test/endpoint")

        mock_delete.assert_called_once_with(
            self.url + "/test/endpoint", headers=self.expected_headers
        )
        self.assertEqual(result, {"test": "data"})

    @patch("requests.delete")
    def test_delete_error(self, mock_delete):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException
        )
        mock_response.text = "Error message"
        mock_delete.return_value = mock_response

        with self.assertRaises(OMMXDA4AdapterError):
            self.client.delete("/test/endpoint")

    @patch.object(DA4Client, "post")
    def test_post_qubo_solve_without_blob(self, mock_post):
        mock_post.return_value = {"job_id": "test-job-id"}

        qubo_request = QuboRequest(
            fujitsuDA3=FujitsuDA3Solver(
                time_limit_sec=1,
                penalty_coef=10000,
                guidance_config={"1": False, "2": False, "4": False},
            ),
            binary_polynomial=BinaryPolynomial(
                terms=[
                    BinaryPolynomialTerm(c=2, p=[1, 2]),
                    BinaryPolynomialTerm(c=-4, p=[2, 4]),
                    BinaryPolynomialTerm(c=-3, p=[]),
                ]
            ),
        )

        result = self.client.post_qubo_solve(qubo_request)

        mock_post.assert_called_once_with(
            "/v4/async/qubo/solve",
            qubo_request.model_dump_json(exclude_none=True, by_alias=True),
            self.expected_headers,
        )
        self.assertEqual(result, "test-job-id")

    @patch.object(DA4Client, "post")
    def test_post_qubo_solve_with_blob(self, mock_post):
        mock_post.return_value = {"job_id": "test-job-id"}

        qubo_request = QuboRequest(
            fujitsuDA3=FujitsuDA3Solver(
                time_limit_sec=1,
                penalty_coef=10000,
                guidance_config={"1": False, "2": False, "4": False},
            ),
            binary_polynomial=BinaryPolynomial(
                terms=[
                    BinaryPolynomialTerm(c=2, p=[1, 2]),
                    BinaryPolynomialTerm(c=-4, p=[2, 4]),
                    BinaryPolynomialTerm(c=-3, p=[]),
                ]
            ),
        )
        blob_sas_token = "test-blob-token"
        blob_account_name = "test-account"

        expected_headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": self.token,
            "X-Blob-Sas-Token": blob_sas_token,
            "X-Storage-Account-Name": blob_account_name,
        }

        result = self.client.post_qubo_solve(
            qubo_request,
            blob_sas_token=blob_sas_token,
            blob_account_name=blob_account_name,
        )

        mock_post.assert_called_once_with(
            "/v4/async/qubo/solve",
            qubo_request.model_dump_json(exclude_none=True, by_alias=True),
            expected_headers,
        )
        self.assertEqual(result, "test-job-id")

    @patch.object(DA4Client, "get")
    def test_get_jobs(self, mock_get):
        mock_get.return_value = {
            "job_status_list": [
                {
                    "job_id": "test-job-id-1",
                    "job_status": "Done",
                    "start_time": "2023-01-01T00:00:00Z",
                },
                {
                    "job_id": "test-job-id-2",
                    "job_status": "Running",
                    "start_time": "2023-01-01T00:01:00Z",
                },
            ]
        }

        result = self.client.get_jobs()

        mock_get.assert_called_once_with("/v4/async/jobs")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["job_id"], "test-job-id-1")
        self.assertEqual(result[1]["job_status"], "Running")

    @patch.object(DA4Client, "get")
    def test_get_job_result(self, mock_get):
        mock_get.return_value = {
            "qubo_solution": "test_qubo_solution",
            "status": "Done",
        }
        job_id = "test-job-id"

        result = self.client.get_job_result(job_id)

        mock_get.assert_called_once_with("/v4/async/jobs/result/test-job-id")
        self.assertEqual(result["status"], "Done")

    @patch.object(DA4Client, "delete")
    def test_delete_job_result_success(self, mock_delete):
        mock_delete.return_value = {
            "qubo_solution": "test_qubo_solution",
            "status": "Deleted",
        }
        job_id = "test-job-id"

        result = self.client.delete_job_result(job_id)

        mock_delete.assert_called_once_with("/v4/async/jobs/result/test-job-id")
        self.assertTrue(result)

    @patch.object(DA4Client, "delete")
    def test_delete_job_result_failure(self, mock_delete):
        mock_delete.side_effect = OMMXDA4AdapterError("Failed to delete")
        job_id = "test-job-id"

        result = self.client.delete_job_result(job_id)

        mock_delete.assert_called_once_with("/v4/async/jobs/result/test-job-id")
        self.assertFalse(result)

    @patch.object(DA4Client, "post")
    def test_post_job_cancel_success(self, mock_post):
        mock_post.return_value = {"status": "Canceled"}
        job_id = "test-job-id"

        result = self.client.post_job_cancel(job_id)

        mock_post.assert_called_once_with(
            "/v4/async/jobs/cancel",
            JobID(job_id=job_id).model_dump_json(),
            self.expected_headers
        )
        self.assertTrue(result)

    @patch.object(DA4Client, "post")
    def test_post_job_cancel_failure(self, mock_post):
        mock_post.side_effect = OMMXDA4AdapterError("Failed to cancel")
        job_id = "test-job-id"

        result = self.client.post_job_cancel(job_id)

        mock_post.assert_called_once_with(
            "/v4/async/jobs/cancel",
            JobID(job_id=job_id).model_dump_json(),
            self.expected_headers,
        )
        self.assertFalse(result)

    @patch.object(DA4Client, "get_jobs")
    @patch.object(DA4Client, "delete_job_result")
    def test_delete_all_jobs(self, mock_delete_job, mock_get_jobs):
        mock_get_jobs.return_value = [
            {"job_id": "test-job-id-1"},
            {"job_id": "test-job-id-2"},
        ]

        self.client.delete_all_jobs()

        mock_get_jobs.assert_called_once()
        mock_delete_job.assert_has_calls([call("test-job-id-1"), call("test-job-id-2")])

    @patch.object(DA4Client, "get_jobs")
    @patch.object(DA4Client, "post_job_cancel")
    def test_cancel_all_jobs(self, mock_cancel_job, mock_get_jobs):
        mock_get_jobs.return_value = [
            {"job_id": "test-job-id-1"},
            {"job_id": "test-job-id-2"},
        ]

        self.client.cancel_all_jobs()

        mock_get_jobs.assert_called_once()
        mock_cancel_job.assert_has_calls([call("test-job-id-1"), call("test-job-id-2")])

    @patch("time.sleep")
    @patch.object(DA4Client, "delete_job_result")
    @patch.object(DA4Client, "get_job_result")
    def test_fetch_job_result_success(self, mock_get_result, mock_delete, mock_sleep):
        job_id = "test-job-id"
        expected_result = {
            "status": "Done",
            "qubo_solution": {
                "progress": [{"energy": 0.0, "penalty_energy": 0.0, "time": 0.0}],
                "result_status": True,
                "solutions": [
                    {
                        "energy": 0.0,
                        "penalty_energy": 0.0,
                        "frequency": 1,
                        "configuration": {"1": True, "2": False},
                    }
                ],
                "timing": {"solve_time": "0.1s", "total_elapsed_time": "0.2s"},
            },
        }
        mock_get_result.return_value = expected_result

        result = self.client.fetch_job_result(job_id)

        mock_get_result.assert_called_once_with(job_id)
        mock_delete.assert_called_once_with(job_id)
        self.assertEqual(result.model_dump(), expected_result)

    @patch.object(DA4Client, "get_job_result")
    @patch("time.sleep")
    def test_fetch_job_result_running_then_done(self, mock_sleep, mock_get_result):
        job_id = "test-job-id"
        running_result = {"status": "Running"}
        done_result = {
            "status": "Done",
            "qubo_solution": {
                "progress": [{"energy": 0.0, "penalty_energy": 0.0, "time": 0.0}],
                "result_status": True,
                "solutions": [
                    {
                        "energy": 0.0,
                        "penalty_energy": 0.0,
                        "frequency": 1,
                        "configuration": {"1": True, "2": False},
                    }
                ],
                "timing": {"solve_time": "0.1s", "total_elapsed_time": "0.2s"},
            },
        }
        mock_get_result.side_effect = [running_result, done_result]

        result = self.client.fetch_job_result(job_id)

        self.assertEqual(mock_get_result.call_count, 2)
        mock_sleep.assert_called_once_with(self.client._solver_access_interval)
        self.assertEqual(result.model_dump(), done_result)

    @patch.object(DA4Client, "get_job_result")
    def test_fetch_job_result_error(self, mock_get_result):
        job_id = "test-job-id"
        error_result = {"status": "Error", "error": "test-error"}
        mock_get_result.return_value = error_result

        with self.assertRaises(OMMXDA4AdapterError):
            self.client.fetch_job_result(job_id)

    @patch.object(DA4Client, "get_job_result")
    def test_fetch_job_result_unknown_status(self, mock_get_result):
        job_id = "test-job-id"
        unknown_result = {"status": "Unknown"}
        mock_get_result.return_value = unknown_result

        with self.assertRaises(OMMXDA4AdapterError):
            self.client.fetch_job_result(job_id)

    @patch.object(DA4Client, "post_qubo_solve")
    @patch.object(DA4Client, "fetch_job_result")
    def test_sample(self, mock_fetch, mock_post_solve):
        job_id = "test-job-id"
        qubo_request = QuboRequest(
            fujitsuDA3=FujitsuDA3Solver(
                time_limit_sec=1,
                penalty_coef=10000,
                guidance_config={"1": False, "2": False, "4": False},
            ),
            binary_polynomial=BinaryPolynomial(
                terms=[
                    BinaryPolynomialTerm(c=2, p=[1, 2]),
                    BinaryPolynomialTerm(c=-4, p=[2, 4]),
                    BinaryPolynomialTerm(c=-3, p=[]),
                ]
            ),
        )
        mock_result = {
            "qubo_solution": {
                "progress": [{"energy": 0.0, "penalty_energy": 0.0, "time": 0.0}],
                "result_status": True,
                "solutions": [
                    {
                        "energy": 0.0,
                        "penalty_energy": 0.0,
                        "frequency": 1,
                        "configuration": {"1": True, "2": False},
                    }
                ],
                "timing": {"solve_time": "0.1s", "total_elapsed_time": "0.2s"},
            },
            "status": "Done",
        }

        mock_post_solve.return_value = job_id
        mock_fetch.return_value = mock_result

        result = self.client.sample(qubo_request)

        mock_post_solve.assert_called_once_with(
            qubo_request=qubo_request, blob_sas_token=None, blob_account_name=None
        )
        mock_fetch.assert_called_once_with(job_id)
        self.assertEqual(result, mock_result)

    @patch.object(DA4Client, "post_qubo_solve")
    @patch.object(DA4Client, "fetch_job_result")
    def test_sample_with_blob(self, mock_fetch, mock_post_solve):
        job_id = "test-job-id"
        qubo_request = QuboRequest(
            fujitsuDA3=FujitsuDA3Solver(
                time_limit_sec=1,
                penalty_coef=10000,
                guidance_config={"1": False, "2": False, "4": False},
            ),
            binary_polynomial=BinaryPolynomial(
                terms=[
                    BinaryPolynomialTerm(c=2, p=[1, 2]),
                    BinaryPolynomialTerm(c=-4, p=[2, 4]),
                    BinaryPolynomialTerm(c=-3, p=[]),
                ]
            ),
        )
        blob_sas_token = "test-blob-token"
        blob_account_name = "test-account"
        mock_result = {
            "qubo_solution": {
                "progress": [{"energy": 0.0, "penalty_energy": 0.0, "time": 0.0}],
                "result_status": True,
                "solutions": [
                    {
                        "energy": 0.0,
                        "penalty_energy": 0.0,
                        "frequency": 1,
                        "configuration": {"1": True, "2": False},
                    }
                ],
                "timing": {"solve_time": "0.1s", "total_elapsed_time": "0.2s"},
            },
            "status": "Done",
        }

        mock_post_solve.return_value = job_id
        mock_fetch.return_value = mock_result

        result = self.client.sample(
            qubo_request,
            blob_sas_token=blob_sas_token,
            blob_account_name=blob_account_name,
        )

        mock_post_solve.assert_called_once_with(
            qubo_request=qubo_request,
            blob_sas_token=blob_sas_token,
            blob_account_name=blob_account_name,
        )
        mock_fetch.assert_called_once_with(job_id)
        self.assertEqual(result, mock_result)


if __name__ == "__main__":
    unittest.main()
