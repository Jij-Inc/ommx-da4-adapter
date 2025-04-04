from enum import Enum
from typing import Literal

import pydantic


class ContentType(str, Enum):
    json = "application/json"


class Accept(str, Enum):
    json = "application/json"


class FujitsuDA3Solver(pydantic.BaseModel):
    time_limit_sec: int = 10
    target_energy: float | None = None
    num_run: int = 16
    num_group: int = 1
    num_output_solution: int = 5
    gs_level: int = 5
    gs_cutoff: int = 8000
    one_hot_level: int = 3
    one_hot_cutoff: int = 100
    internal_penalty: int = 0
    penalty_auto_mode: int = 1
    penalty_coef: int = 1
    penalty_inc_rate: int = 150
    max_penalty_coef: int = 0
    guidance_config: dict[str, bool] = {}
    fixed_config: dict[str, bool] = {}
    one_way_one_hot_groups: dict[Literal["numbers"], list[int]] = {}
    two_way_one_hot_groups: dict[Literal["numbers"], list[int]] = {}

    @pydantic.field_validator("time_limit_sec")
    @classmethod
    def check_time_limit_sec(cls, v):
        if not (1 <= v <= 3600):
            raise ValueError("`time_limit_sec` must be between 1 and 3600")
        return v

    @pydantic.field_validator("target_energy")
    @classmethod
    def check_target_energy(cls, v):
        if v is None:
            return v

        # NOTE: Limited to a range where no numerical errors occur in DA4.
        if not (-pow(2, 52) <= v <= pow(2, 52)):
            raise ValueError("`target_energy` must be between - 2^52 and 2^52")
        return v

    @pydantic.field_validator("num_run")
    @classmethod
    def check_num_run(cls, v):
        if not (1 <= v <= 1024):
            raise ValueError("`num_run` must be between 1 and 1024")
        return v

    @pydantic.field_validator("num_group")
    @classmethod
    def check_num_group(cls, v):
        if not (1 <= v <= 16):
            raise ValueError("`num_group` must be between 1 and 16")
        return v

    @pydantic.field_validator("num_output_solution")
    @classmethod
    def check_num_output_solution(cls, v):
        if not (1 <= v <= 1024):
            raise ValueError("`num_output_solution` must be between 1 and 1024")
        return v

    @pydantic.field_validator("gs_level")
    @classmethod
    def check_gs_level(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("`gs_level` must be between 0 and 100")
        return v

    @pydantic.field_validator("gs_cutoff")
    @classmethod
    def check_gs_cutoff(cls, v):
        if not (0 <= v <= 1000000):
            raise ValueError("`gs_cutoff` must be between 0 and 1000000")
        return v

    @pydantic.field_validator("one_hot_level")
    @classmethod
    def check_one_hot_level(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("`one_hot_level` must be between 0 and 100")
        return v

    @pydantic.field_validator("one_hot_cutoff")
    @classmethod
    def check_one_hot_cutoff(cls, v):
        if not (0 <= v <= 1000000):
            raise ValueError("`one_hot_cutoff` must be between 0 and 1000000")
        return v

    @pydantic.field_validator("internal_penalty")
    @classmethod
    def check_internal_penalty(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("`internal_penalty` must be 0 or 1")
        return v

    @pydantic.field_validator("penalty_auto_mode")
    @classmethod
    def check_penalty_auto_mode(cls, v):
        if not (0 <= v <= 10000):
            raise ValueError("`penalty_auto_mode` must be between 0 and 10000")
        return v

    @pydantic.field_validator("penalty_coef")
    @classmethod
    def check_penalty_coef(cls, v):
        if not (1 <= v <= 9223372036854775807):
            raise ValueError("`penalty_coef` must be between 1 and 9223372036854775807")
        return v

    @pydantic.field_validator("penalty_inc_rate")
    @classmethod
    def check_penalty_inc_rate(cls, v):
        if not (100 <= v <= 200):
            raise ValueError("`penalty_inc_rate` must be between 100 and 200")
        return v

    @pydantic.field_validator("max_penalty_coef")
    @classmethod
    def check_max_penalty_coef(cls, v):
        if not (0 <= v <= 9223372036854775807):
            raise ValueError(
                "`max_penalty_coef` must be between 0 and 9223372036854775807"
            )
        return v


class BinaryPolynomialTerm(pydantic.BaseModel):
    """
    NOTE:
    In DA4, `coefficient` and `polynomials` can be used instead of `c` and `p`.
    However, since `c` and `p` are used with priority, we limit the attributes to `c` and `p` here.
    """

    c: float
    p: list[int]

    @pydantic.field_validator("c")
    @classmethod
    def check_c(cls, v):
        # NOTE: Limited to a range where no numerical errors occur in DA4.
        if not (-pow(2, 52) <= v <= pow(2, 52)):
            raise ValueError("`c` must be between - 2^52 and 2^52")
        return v


class BinaryPolynomial(pydantic.BaseModel):
    terms: list[BinaryPolynomialTerm]


class PenaltyBinaryPolynomial(pydantic.BaseModel):
    terms: list[BinaryPolynomialTerm]


class Inequality(pydantic.BaseModel):
    terms: list[BinaryPolynomialTerm]
    lambda_: int = pydantic.Field(alias="lambda", default=1)

    @pydantic.field_validator("lambda_")
    @classmethod
    def check_lambda_(cls, v):
        if not (1 <= v <= 1000000000):
            raise ValueError("`lambda` must be between 1 and 1000000000 ")
        return v


class QuboRequest(pydantic.BaseModel):
    fujitsuDA3: FujitsuDA3Solver
    binary_polynomial: BinaryPolynomial | None = None
    penalty_binary_polynomial: PenaltyBinaryPolynomial | None = None
    inequalities: list[Inequality] | None = None
    object_format: str | None = None
    bucket_name: str | None = None
    binary_polynomial_object_name: str | None = None
    penalty_binary_polynomial_object_name: str | None = None
    inequalities_object_name: str | None = None


class Progress(pydantic.BaseModel):
    energy: float
    penalty_energy: float
    time: float


class QuboSolution(pydantic.BaseModel):
    energy: float
    penalty_energy: float
    frequency: int
    configuration: dict[str, bool]


class SolverTiming(pydantic.BaseModel):
    solve_time: str
    total_elapsed_time: str


class QuboSolutionList(pydantic.BaseModel):
    progress: list[Progress]
    result_status: bool
    solutions: list[QuboSolution]
    timing: SolverTiming


class QuboResponse(pydantic.BaseModel):
    qubo_solution: QuboSolutionList
    status: Literal["Done", "Deleted"]


class QuboErrorResponse(pydantic.BaseModel):
    message: str
    status: Literal["Done", "Deleted"]


class JobID(pydantic.BaseModel):
    job_id: str


class JobStatus(pydantic.BaseModel):
    status: Literal["Waiting", "Running", "Done", "Canceled", "Error"]


class JobStatusInfo(pydantic.BaseModel):
    job_id: str
    job_status: Literal["Waiting", "Running", "Done", "Canceled", "Error"]
    start_time: str


class JobStatusList(pydantic.BaseModel):
    job_status_list: list[JobStatusInfo]
