from typing import List, Literal, Optional, Union

from ommx.adapter import SamplerAdapter
from ommx.v1 import (
    Instance,
    SampleSet,
    Constraint,
    DecisionVariable,
    State,
    Samples,
)

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


class OMMXDA4Adapter(SamplerAdapter):
    def __init__(
        self,
        ommx_instance: Instance,
        *,
        time_limit_sec: int = 10,
        target_energy: Optional[float] = None,
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
        guidance_config: Optional[dict[str, bool]] = None,
        fixed_config: Optional[dict[str, bool]] = None,
        inequalities_lambda: Optional[dict[int, int]] = None,
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
        self._ommx_instance = ommx_instance
        self._inequalities_lambda = inequalities_lambda

        self._check_decision_variable()
        self._one_hot_dict = self._generate_one_hot_dict()
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

    @classmethod
    def sample(
        cls,
        ommx_instance: Instance,
        *,
        token: Optional[str] = None,
        url: str = "https://api.aispf.global.fujitsu.com/da",
        version: Literal["v4", "v3c"] = "v4",
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

        adapter = cls(ommx_instance)
        qubo_request = adapter.sampler_input
        client = DA4Client(token=token, url=url, version=version)
        qubo_response = client.sample(qubo_request=qubo_request)

        return adapter.decode_to_sampleset(qubo_response)

    def decode_to_sampleset(self, data: QuboResponse) -> SampleSet:
        """Decode QuboResponse to SampleSet.

        :param data: The QUBO result data from DA4
        :return: SampleSet
        """

        sample_id: int = 0
        entries: List[Samples.SamplesEntry] = []

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
            entries.append(
                Samples.SamplesEntry(
                    state=State(entries=converted_configuration),
                    ids=[i for i in range(sample_id, next_sample_id)],
                )
            )
            sample_id = next_sample_id

        samples = Samples(
            entries=entries,
        )

        return self._ommx_instance.evaluate_samples(samples)

    def _check_decision_variable(self):
        """Check if the decision variables are binary."""
        instance = self._ommx_instance

        for decision_variable in instance.get_decision_variables():
            if decision_variable.kind != DecisionVariable.BINARY:
                raise OMMXDA4AdapterError(
                    f"The decision variable must be binary: id {decision_variable.id}"
                )

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
                c=value, p=self._replace_polynomials_with_variable_map(tuple(key))
            )
            for key, value in terms.items()
        ]

        return BinaryPolynomial(terms=binary_polynomial_terms)

    def _generate_penalty_binary_polynomial(
        self,
    ) -> Union[PenaltyBinaryPolynomial, None]:
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
        squared_terms_dict = {}
        for constraint in instance.get_constraints():
            # skip if not equality constraints
            if constraint.equality != Constraint.EQUAL_TO_ZERO:
                continue

            if constraint.id in self._one_hot_dict:
                continue

            function = constraint.function
            squared_function = function * function

            for key, value in squared_function.terms.items():
                # Remove duplicates by converting to set
                set_key = set(key)
                list_key = tuple(set_key)

                # Add to existing value when duplicates occur
                if list_key in squared_terms_dict:
                    squared_terms_dict[list_key] += value
                else:
                    squared_terms_dict[list_key] = value

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

    def _generate_inequalities(self) -> Union[list[Inequalities], None]:
        """Generate Inequalities from OMMX instance.

        :return: Inequalities
        """
        instance = self._ommx_instance

        inequalities_list = []
        for constraint in instance.get_constraints():
            # skip if not inequality constraints
            if constraint.equality != Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                continue

            terms = constraint.function.terms
            inequalities_terms = [
                BinaryPolynomialTerm(
                    c=value, p=self._replace_polynomials_with_variable_map(tuple(key))
                )
                for key, value in terms.items()
            ]

            if (
                self._inequalities_lambda is None
                or constraint.id not in self._inequalities_lambda
            ):
                lambda_ = 1
            else:
                lambda_ = self._inequalities_lambda[constraint.id]
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
        instance = self._ommx_instance

        if len(instance.raw.constraint_hints.one_hot_constraints) == 0:
            return 0
        else:
            return 1

    def _generate_variable_map(self) -> dict[int, int]:
        """Generate variable map that represents the correspondence
        between the IDs of decision variables in ommx.v1.Instance and the variable numbers on QuboRequest.
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
        for decision_variable in instance.get_decision_variables():
            # skip if already in variable_map
            if decision_variable.id in variable_map:
                continue
            variable_map[decision_variable.id] = index
            index += 1

        return variable_map

    def _replace_polynomials_with_variable_map(
        self, polynomial: tuple[int, ...]
    ) -> list[int]:
        """Replace the IDs of decision variables in ommx.v1.Instance
        with variable numbers on QuboRequest using variable map.

        variable map is generated by _generate_variable_map().
        This corresponds between the IDs of decision variables in ommx.v1.Instance
        and the variable numbers on QuboRequest.
        """
        transformed_polynomial = [self._variable_map[p] for p in polynomial]
        return transformed_polynomial

    def _generate_one_way_one_hot_groups(
        self,
    ) -> Union[dict[Literal["numbers"], list[int]], None]:
        """Generate one way one hot groups."""

        numbers = [len(variables) for variables in self._one_hot_dict.values()]

        if len(numbers) == 0:
            return None
        else:
            return {"numbers": numbers}

    def _generate_one_hot_dict(self) -> dict[int, list[int]]:
        """Generate a dictionary of one-hot constraints without duplicate decision variables.

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

        case 2：duplicate decision variables
        Prioritize longer constraints and exclude shorter ones
        constraint_1: id=0, x₀ + x₁ + x₂ = 1
        constraint_2: id=1, x₁ + x₃ = 1
        one_hot_dict = {
                            0: [0, 1, 2]
                        }
        """
        instance = self._ommx_instance

        sorted_one_hot_constraints = sorted(
            instance.raw.constraint_hints.one_hot_constraints,
            key=lambda x: len(x.variables),
            reverse=True,
        )

        one_hot_dict: dict[int, list[int]] = {}
        used_variables = set()
        for one_hot_constraint in sorted_one_hot_constraints:
            if any(
                [
                    variable in used_variables
                    for variable in one_hot_constraint.variables
                ]
            ):
                continue

            used_variables.update(one_hot_constraint.variables)
            one_hot_dict[one_hot_constraint.id] = [
                variable_id for variable_id in one_hot_constraint.variables
            ]

        return one_hot_dict
