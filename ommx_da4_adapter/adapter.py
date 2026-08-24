from typing import ClassVar, Literal

from ommx import (
    Constraint,
    DegreeBound,
    Equality,
    Instance,
    InstanceClass,
    InstanceClassClause,
    Kind,
    PreparationPolicy,
    Samples,
    SampleSet,
    Sense,
    Solution,
    SpecialConstraintKind,
    SpecialConstraintPreparation,
    State,
)
from ommx.adapter import DiagnosticsSink, SamplerAdapter

from .client import DA4Client
from .exception import OMMXDA4AdapterError
from .models import (
    BinaryPolynomial,
    BinaryPolynomialTerm,
    FujitsuDA3Solver,
    Inequalities,
    PenaltyBinaryPolynomial,
    QuboRequest,
    QuboResponse,
)

_UNBOUNDED_REGULAR_CONSTRAINT_DEGREE_BOUNDS = {
    Equality.EqualToZero: DegreeBound.unbounded(),
    Equality.LessThanOrEqualToZero: DegreeBound.unbounded(),
}


class OMMXDA4Adapter(SamplerAdapter):
    INPUT_CLASS: ClassVar[InstanceClass | None] = InstanceClass(
        [
            InstanceClassClause(
                label="da4-binary-polynomial-with-one-hot",
                allowed_variable_kinds={Kind.Binary},
                objective_degree_bound=DegreeBound.unbounded(),
                regular_constraint_degree_bounds=(
                    _UNBOUNDED_REGULAR_CONSTRAINT_DEGREE_BOUNDS
                ),
                allows_one_hot=True,
                allowed_senses={Sense.Minimize, Sense.Maximize},
            )
        ]
    )

    @classmethod
    def recommended_preparation_policy(cls) -> PreparationPolicy:
        """Recommend lowering unsupported special constraints before using DA4.

        DA4 accepts OneHot constraints directly through one-way one-hot groups.
        This recommendation therefore preserves OneHot constraints and lowers
        only Indicator and SOS1 constraints. The returned policy is fresh and
        caller-editable.
        """
        return PreparationPolicy(
            special_constraints=SpecialConstraintPreparation.lower_special_constraints(
                kinds={
                    SpecialConstraintKind.Indicator,
                    SpecialConstraintKind.Sos1,
                }
            )
        )

    def __init__(
        self,
        ommx_instance: Instance,
        *,
        time_limit_sec: int = 10,
        target_energy: float | None = None,
        num_run: int = 16,
        num_group: int = 1,
        num_output_solution: int = 5,
        gs_level: int = 5,
        gs_cutoff: int = 8000,
        one_hot_level: int = 3,
        one_hot_cutoff: int = 100,
        penalty_auto_mode: int = 1,
        penalty_coef: int = 1,
        penalty_inc_rate: int = 150,
        max_penalty_coef: int = 0,
        guidance_config: dict[str, bool] | None = None,
        fixed_config: dict[str, bool] | None = None,
        inequalities_lambda: dict[int, int] | None = None,
    ):
        """Digital Annealer adapter for OMMX.

        :param ommx_instance: OMMX instance
        :param time_limit_sec: Upper limit of execution time (in seconds)
        :param target_energy: Target energy to be terminated when reached
        :param num_run: Number of parallel trial iterations
        :param num_group: Number of groups of parallel trials
        :param num_output_solution: Number of output solutions for each parallel trial group
        :param gs_level: Levels of global search
        :param gs_cutoff: Convergence decision level for global search. If 0 is set, this function is turned off
        :param one_hot_level: Levels of 1hot constraints search
        :param one_hot_cutoff: Convergence decision level for 1hot constraints search. If 0 is set, this function is turned off
        :param penalty_auto_mode: Coefficient adjustment mode for constraint terms. If set to 0, no adjustment is made
        :param penalty_coef: Coefficient of the constraint term
        :param penalty_inc_rate: Parameters for automatic adjustment of constraint terms
        :param max_penalty_coef: Maximum value of constraint term coefficients. If 0 is set, there is no maximum value
        :param guidance_config: Initial value of each variable
        :param fixed_config: Fixed value for each variable
        :param inequalities_lambda: Coefficient of inequality. If omitted, set to 1. Defaults to None.
        """
        self.require_applicable(ommx_instance)

        self._ommx_instance = ommx_instance
        self._inequalities_lambda = inequalities_lambda

        (
            self._one_hot_dict,
            self._penalty_one_hot_dict,
        ) = self._partition_one_hot_constraints()
        self._variable_map = self._generate_variable_map()

        # generate QuboRequest from OMMX instance
        self._sampler_input = QuboRequest(
            fujitsuDA3=FujitsuDA3Solver(
                time_limit_sec=time_limit_sec,
                target_energy=target_energy,
                num_run=num_run,
                num_group=num_group,
                num_output_solution=num_output_solution,
                gs_level=gs_level,
                gs_cutoff=gs_cutoff,
                one_hot_level=one_hot_level,
                one_hot_cutoff=one_hot_cutoff,
                internal_penalty=self._generate_internal_penalty(),
                penalty_auto_mode=penalty_auto_mode,
                penalty_coef=penalty_coef,
                penalty_inc_rate=penalty_inc_rate,
                max_penalty_coef=max_penalty_coef,
                guidance_config=guidance_config,
                fixed_config=fixed_config,
                one_way_one_hot_groups=self._generate_one_way_one_hot_groups(),
            ),
            binary_polynomial=self._generate_binary_polynomial(),
            penalty_binary_polynomial=self._generate_penalty_binary_polynomial(),
            inequalities=self._generate_inequalities(),
        )

    @property
    def sampler_input(self) -> QuboRequest:
        """Get QuboRequest from OMMX instance.

        :return: QuboRequest
        """
        return self._sampler_input

    @property
    def solver_input(self) -> QuboRequest:
        """Get QuboRequest from OMMX instance.

        :return: QuboRequest
        """
        return self._sampler_input

    @classmethod
    def sample(
        cls,
        ommx_instance: Instance,
        *,
        token: str | None = None,
        url: str = "https://api.aispf.global.fujitsu.com/da",
        version: Literal["v4", "v3c"] = "v4",
        diagnostics: DiagnosticsSink | None = None,
    ) -> SampleSet:
        """Sample the result in DA4 with DA4Client.

        :param ommx_instance: OMMX instance
        :param token: Authentication token for DA4 API. Defaults to None.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :return: SampleSet
        """
        if token is None:
            raise OMMXDA4AdapterError(
                "token is required. Please set the token to use the DA4 API."
            )

        _ = diagnostics
        adapter = cls(ommx_instance)
        qubo_request = adapter.sampler_input
        client = DA4Client(token=token, url=url, version=version)
        qubo_response = client.sample(qubo_request=qubo_request)

        return adapter.decode_to_sampleset(qubo_response)

    @classmethod
    def solve(
        cls,
        ommx_instance: Instance,
        *,
        token: str | None = None,
        url: str = "https://api.aispf.global.fujitsu.com/da",
        version: Literal["v4", "v3c"] = "v4",
        diagnostics: DiagnosticsSink | None = None,
    ) -> Solution:
        """Solve the result in DA4 with DA4Client.

        :param ommx_instance: OMMX instance
        :param token: Authentication token for DA4 API. Defaults to None.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :return: Solution
        """
        sample_set = cls.sample(
            ommx_instance,
            token=token,
            url=url,
            version=version,
            diagnostics=diagnostics,
        )
        return sample_set.best_feasible

    def decode_to_sampleset(self, data: QuboResponse) -> SampleSet:
        """Decode QuboResponse to SampleSet.

        :param data: The QUBO result data from DA4
        :return: SampleSet
        """

        sample_id: int = 0
        samples: Samples = Samples({})  # Create empty samples

        reversed_variable_map = {v: k for k, v in self._variable_map.items()}

        for solution in data.qubo_solution.solutions:
            configuration = solution.configuration
            try:
                converted_configuration = {
                    reversed_variable_map[int(k)]: int(v)
                    for k, v in configuration.items()
                }
            except KeyError as e:
                raise OMMXDA4AdapterError(
                    f"Invalid solution configuration: The solution contains an unexpected decision variable id ({e})."
                )
            next_sample_id = sample_id + solution.frequency
            samples.append(
                sample_ids=[i for i in range(sample_id, next_sample_id)],
                state=State(entries=converted_configuration),
            )
            sample_id = next_sample_id

        return self._ommx_instance.evaluate_samples(samples)

    def decode(self, data: QuboResponse) -> Solution:
        """Decode QuboResponse to Solution.

        :param data: The QUBO result data from DA4
        :return: Solution
        """
        sample_set = self.decode_to_sampleset(data)
        return sample_set.best_feasible

    def _generate_binary_polynomial(self) -> BinaryPolynomial:
        """Generate BinaryPolynomial from OMMX instance."

        :return: BinaryPolynomial
        """
        instance = self._ommx_instance

        function = instance.objective

        # if sense is maximize, multiply by -1 (DA4 only supports minimization)
        if instance.sense == Instance.MAXIMIZE:
            function = -instance.objective

        # get objective terms
        terms = function.terms

        binary_polynomial_terms = [
            BinaryPolynomialTerm(
                c=value, p=self._replace_polynomials_with_variable_map(key)
            )
            for key, value in terms.items()
        ]

        return BinaryPolynomial(terms=binary_polynomial_terms)

    def _generate_penalty_binary_polynomial(
        self,
    ) -> PenaltyBinaryPolynomial | None:
        """Generate PenaltyBinaryPolynomial from OMMX instance.

        Example:
        =========
        Original: x₀ + x₁ - 1
        Squared: (x₀ + x₁ - 1)² = x₀² + 2x₀x₁ - 2x₀ + x₁² - 2x₁ + 1
        After binary simplification: x₀ + 2x₀x₁ - 2x₀ + x₁ - 2x₁ + 1 = 2x₀x₁ - x₀ - x₁ + 1
        Dictionary form: {(0, 1): 2.0, (0,): -1.0, (1,): -1.0, (): 1.0}

        :return: PenaltyBinaryPolynomial
        """
        instance = self._ommx_instance

        # Squared Polynomial with Binary Variables
        squared_terms_dict: dict[tuple[int, ...], float] = {}

        def add_term(key: tuple[int, ...], value: float) -> None:
            # Binary variables satisfy x^n = x for every positive integer n.
            binary_key = tuple(sorted(set(key)))
            squared_terms_dict[binary_key] = (
                squared_terms_dict.get(binary_key, 0.0) + value
            )

        for constraint in instance.constraints.values():
            # skip if not equality constraints
            if constraint.equality != Constraint.EQUAL_TO_ZERO:
                continue

            function = constraint.function
            squared_function = function * function

            for key, value in squared_function.terms.items():
                add_term(key, value)

        # DA4 one-way one-hot groups cannot share decision variables. Treat each
        # overlapping group that was not selected for native handling as the
        # regular equality sum(x_i) - 1 = 0. For binary variables, its square is
        # 2 * sum_{i < j}(x_i * x_j) - sum_i(x_i) + 1.
        for variables in self._penalty_one_hot_dict.values():
            add_term((), 1.0)
            for variable in variables:
                add_term((variable,), -1.0)
            for index, left in enumerate(variables):
                for right in variables[index + 1 :]:
                    add_term((left, right), 2.0)

        penalty_binary_polynomial_terms = [
            BinaryPolynomialTerm(
                c=value, p=self._replace_polynomials_with_variable_map(key)
            )
            for key, value in squared_terms_dict.items()
        ]

        if len(penalty_binary_polynomial_terms) == 0:
            return None
        else:
            return PenaltyBinaryPolynomial(terms=penalty_binary_polynomial_terms)

    def _generate_inequalities(self) -> list[Inequalities] | None:
        """Generate Inequalities from OMMX instance.

        :return: Inequalities
        """
        instance = self._ommx_instance

        inequalities_list = []
        for constraint_id, constraint in instance.constraints.items():
            # skip if not inequality constraints
            if constraint.equality != Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                continue

            terms = constraint.function.terms
            inequalities_terms = [
                BinaryPolynomialTerm(
                    c=value, p=self._replace_polynomials_with_variable_map(key)
                )
                for key, value in terms.items()
            ]

            if (
                self._inequalities_lambda is None
                or constraint_id not in self._inequalities_lambda
            ):
                lambda_ = 1
            else:
                lambda_ = self._inequalities_lambda[constraint_id]
            inequalities_list.append(
                Inequalities(terms=inequalities_terms, **{"lambda": lambda_})
            )

        if len(inequalities_list) == 0:
            return None
        else:
            return inequalities_list

    def _generate_internal_penalty(self) -> int:
        """Generate internal penalty

        if set one way one hot or two way one hot, internal_penalty is 1
        else, internal_penalty is 0

        caution: two way one hot is not supported yet
        """
        return int(bool(self._one_hot_dict))

    def _generate_variable_map(self) -> dict[int, int]:
        """Generate variable map that represents the correspondence
        between the IDs of decision variables in ommx.Instance and the variable numbers on QuboRequest.
        """
        instance = self._ommx_instance
        variable_map = {}

        # First enumerate the decision variables from one-hot constraints,
        # then enumerate the remaining decision variables afterwards.
        index = 0

        for variables in self._one_hot_dict.values():
            for variable in variables:
                variable_map[variable] = index
                index += 1
        for decision_variable in instance.used_decision_variables:
            # skip if already in variable_map
            if decision_variable.id in variable_map:
                continue
            variable_map[decision_variable.id] = index
            index += 1

        return variable_map

    def _replace_polynomials_with_variable_map(
        self, polynomial: tuple[int, ...]
    ) -> list[int]:
        """Replace the IDs of decision variables in ommx.Instance
        with variable numbers on QuboRequest using variable map.

        variable map is generated by _generate_variable_map().
        This corresponds between the IDs of decision variables in ommx.Instance
        and the variable numbers on QuboRequest.
        """
        transformed_polynomial = [self._variable_map[p] for p in polynomial]
        return transformed_polynomial

    def _generate_one_way_one_hot_groups(
        self,
    ) -> dict[Literal["numbers"], list[int]] | None:
        """Generate one way one hot groups."""

        numbers = [len(variables) for variables in self._one_hot_dict.values()]

        if len(numbers) == 0:
            return None
        else:
            return {"numbers": numbers}

    def _partition_one_hot_constraints(
        self,
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Partition one-hot constraints for native and penalty handling.

        Examples:
        =========
        case 1：no duplicate decision variables
        constraint_1: id=0, x₀ + x₁ + x₂ = 1
        constraint_2: id=1, x₃ + x₄ = 1
        constraint_3: id=2, x₅ + x₆ = 1
        one_hot_dict = {
                           0: [0, 1, 2],
                           1: [3, 4],
                           2: [5, 6],
                       }
        penalty_one_hot_dict = {}

        case 2：duplicate decision variables
        Prioritize longer constraints for native handling and send shorter ones
        to penalty handling
        constraint_1: id=0, x₀ + x₁ + x₂ = 1
        constraint_2: id=1, x₁ + x₃ = 1
        one_hot_dict = {
                           0: [0, 1, 2]
                       }
        penalty_one_hot_dict = {
                                   1: [1, 3]
                               }
        """
        instance = self._ommx_instance

        sorted_one_hot_constraints = sorted(
            instance.one_hot_constraints.items(),
            key=lambda item: len(item[1].variables),
            reverse=True,
        )

        one_hot_dict: dict[int, list[int]] = {}
        penalty_one_hot_dict: dict[int, list[int]] = {}
        used_variables: set[int] = set()
        for constraint_id, one_hot_constraint in sorted_one_hot_constraints:
            variables = list(one_hot_constraint.variables)

            if used_variables.intersection(variables):
                penalty_one_hot_dict[constraint_id] = variables
                continue

            used_variables.update(variables)
            one_hot_dict[constraint_id] = variables

        return one_hot_dict, penalty_one_hot_dict
