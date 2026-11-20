import copy
from collections.abc import Mapping
from typing import ClassVar, Literal

from ommx import (
    Constraint,
    ConstraintID,
    ConstraintKind,
    DecisionVariable,
    Equality,
    Function,
    Instance,
    InstanceClass,
    InstanceClassClause,
    Kind,
    Linear,
    OneHotConstraintID,
    PolynomialRequirement,
    PreparationPolicy,
    Samples,
    SampleSet,
    Sense,
    Solution,
    SpecialConstraintPreparation,
    get_default_atol,
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


class OMMXDA4Adapter(SamplerAdapter):
    INPUT_CLASS: ClassVar[InstanceClass] = InstanceClass(
        [
            InstanceClassClause(
                label="da4-binary-polynomial-with-one-hot",
                allowed_variable_kinds={Kind.Binary},
                objective_polynomial_requirement=PolynomialRequirement.at_most(2),
                regular_constraint_polynomial_requirements={
                    # Some quadratic equalities yield quadratic penalties after
                    # squaring and binary simplification, e.g. xy = 0: (xy)^2 = xy.
                    # _generate_penalty_binary_polynomial() checks the final degree
                    # after squaring, binary simplification, and aggregation.
                    Equality.EqualToZero: PolynomialRequirement.at_most(2),
                    Equality.LessThanOrEqualToZero: PolynomialRequirement.at_most(1),
                },
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
                    ConstraintKind.Indicator,
                    ConstraintKind.Sos1,
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
        inequalities_lambda: Mapping[int, int]
        | Mapping[ConstraintID, int]
        | Mapping[int | ConstraintID, int]
        | None = None,
    ):
        """Digital Annealer adapter for OMMX.

        Keep the supplied instance unchanged until decoding is complete.

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
        :param inequalities_lambda: Coefficients keyed by regular constraint ID or its integer value. The default coefficient is 1.
        """
        self.require_applicable(ommx_instance)

        self._ommx_instance = ommx_instance
        self._inequalities_lambda = None
        if inequalities_lambda is not None:
            self._inequalities_lambda = {
                ConstraintID(key) if isinstance(key, int) else key: value
                for key, value in inequalities_lambda.items()
            }
            if len(self._inequalities_lambda) != len(inequalities_lambda):
                raise OMMXDA4AdapterError(
                    "Duplicate constraint ID in inequalities_lambda."
                )
        self._validate_constraints()

        (
            self._one_hot_dict,
            self._penalty_one_hot_dict,
        ) = self._partition_one_hot_constraints()
        variables = ommx_instance.active_decision_variables
        self._variable_map = self._generate_variable_map(variables)

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
            inequalities=self._generate_inequalities(variables),
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
        """Sample the given ommx.Instance using DA4Client, returning the samples
        as an ommx.SampleSet.

        ``diagnostics`` are not available through this Adapter.
        The reserved ``diagnostics`` argument is accepted for compatibility with
        the OMMX SamplerAdapter interface.

        **NOTE** The ``token`` parameter *must* be passed to properly
          instantiate the DA4Client. Using the default value will result in an
          error.

        :param ommx_instance: The ommx.Instance to prepare and sample.
        :param token: Authentication token for DA4 API. Defaults to None.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :param diagnostics: Reserved for OMMX SamplerAdapter compatibility;
          currently unused.
        :return: SampleSet
        """
        prepared = copy.copy(ommx_instance)
        prepared.prepare(
            cls.INPUT_CLASS,
            cls.recommended_preparation_policy(),
        )
        return cls.sample_without_preparation(
            prepared,
            token=token,
            url=url,
            version=version,
            diagnostics=diagnostics,
        )

    @classmethod
    def sample_without_preparation(
        cls,
        ommx_instance: Instance,
        *,
        token: str | None = None,
        url: str = "https://api.aispf.global.fujitsu.com/da",
        version: Literal["v4", "v3c"] = "v4",
        diagnostics: DiagnosticsSink | None = None,
    ) -> SampleSet:
        """Sample an exact DA4 Adapter input without preparing it.

        Use this method when the input instance has already been prepared,
        possibly with a custom policy, or already belongs to ``INPUT_CLASS``.

        ``diagnostics`` are not available through this Adapter.
        The reserved ``diagnostics`` argument is accepted for compatibility with
        the OMMX SamplerAdapter interface.

        **NOTE** The ``token`` parameter *must* be passed to properly
          instantiate the DA4Client. Using the default value will result in an
          error.

        :param ommx_instance: The exact DA4 Adapter input to sample.
        :param token: Authentication token for DA4 API. Defaults to None.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :param diagnostics: Reserved for OMMX SamplerAdapter compatibility;
          currently unused.
        :return: SampleSet
        """
        adapter, response = cls._sample_da4(ommx_instance, token, url, version)
        return adapter.decode_to_sampleset(response)

    @classmethod
    def _sample_da4(
        cls,
        ommx_instance: Instance,
        token: str | None,
        url: str,
        version: Literal["v4", "v3c"],
    ) -> tuple["OMMXDA4Adapter", QuboResponse]:
        if token is None:
            raise OMMXDA4AdapterError(
                "token is required. Please set the token to use the DA4 API."
            )

        adapter = cls(ommx_instance)
        client = DA4Client(token=token, url=url, version=version)
        return adapter, client.sample(qubo_request=adapter.sampler_input)

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
        """Solve the given ommx.Instance using DA4Client, returning the best
        feasible solution as an ommx.Solution.

        ``diagnostics`` are not available through this Adapter.
        The reserved ``diagnostics`` argument is accepted for compatibility with
        the OMMX SamplerAdapter interface.

        **NOTE** The ``token`` parameter *must* be passed to properly
          instantiate the DA4Client. Using the default value will result in an
          error.

        :param ommx_instance: The ommx.Instance to prepare and solve.
        :param token: Authentication token for DA4 API. Defaults to None.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :param diagnostics: Reserved for OMMX SamplerAdapter compatibility;
          currently unused.
        :return: Solution
        """
        prepared = copy.copy(ommx_instance)
        prepared.prepare(
            cls.INPUT_CLASS,
            cls.recommended_preparation_policy(),
        )
        return cls.solve_without_preparation(
            prepared,
            token=token,
            url=url,
            version=version,
            diagnostics=diagnostics,
        )

    @classmethod
    def solve_without_preparation(
        cls,
        ommx_instance: Instance,
        *,
        token: str | None = None,
        url: str = "https://api.aispf.global.fujitsu.com/da",
        version: Literal["v4", "v3c"] = "v4",
        diagnostics: DiagnosticsSink | None = None,
    ) -> Solution:
        """Solve an exact DA4 Adapter input without preparing it.

        Use this method when the input instance has already been prepared,
        possibly with a custom policy, or already belongs to ``INPUT_CLASS``.

        ``diagnostics`` are not available through this Adapter.
        The reserved ``diagnostics`` argument is accepted for compatibility with
        the OMMX SamplerAdapter interface.

        **NOTE** The ``token`` parameter *must* be passed to properly
          instantiate the DA4Client. Using the default value will result in an
          error.

        :param ommx_instance: The exact DA4 Adapter input to solve.
        :param token: Authentication token for DA4 API. Defaults to None.
        :param url: URL to the Fujitsu Digital Annealer. Defaults to "https://api.aispf.global.fujitsu.com/da".
        :param version: The version of Digital Annealer as either "v4" or "v3c". Defaults to "v4".
        :param diagnostics: Reserved for OMMX SamplerAdapter compatibility;
          currently unused.
        :return: Solution
        """
        adapter, response = cls._sample_da4(ommx_instance, token, url, version)
        return adapter.decode(response)

    def decode_to_sampleset(self, data: QuboResponse) -> SampleSet:
        """Decode QuboResponse to SampleSet.

        :param data: The QUBO result data from DA4
        :return: SampleSet
        """

        return self._evaluate_samples(data, expand_frequency=True)

    def _evaluate_samples(
        self, data: QuboResponse, *, expand_frequency: bool
    ) -> SampleSet:
        sample_id = 0
        samples = Samples({})

        reversed_variable_map = {v: k for k, v in self._variable_map.items()}

        for solution in data.qubo_solution.solutions:
            if solution.frequency == 0:
                continue
            # Initialize all mapped variables to zero for each solution, then
            # overwrite them with the values returned by DA4. This also supplies
            # values needed by OMMX for variables canceled from the penalty.
            converted_configuration = dict.fromkeys(self._variable_map, 0)
            configuration = solution.configuration
            try:
                converted_configuration.update(
                    {
                        reversed_variable_map[int(k)]: int(v)
                        for k, v in configuration.items()
                    }
                )
            except KeyError as e:
                raise OMMXDA4AdapterError(
                    f"Invalid solution configuration: The solution contains an unexpected decision variable id ({e})."
                )
            next_sample_id = sample_id + (solution.frequency if expand_frequency else 1)
            samples.append(range(sample_id, next_sample_id), converted_configuration)
            sample_id = next_sample_id

        return self._ommx_instance.evaluate_samples(samples)

    def decode(self, data: QuboResponse) -> Solution:
        """Decode QuboResponse to Solution.

        :param data: The QUBO result data from DA4
        :return: Solution
        """
        return self._evaluate_samples(data, expand_frequency=False).best_feasible

    def _validate_constraints(self) -> None:
        """Validate regular constraints and reject infeasible constants."""
        atol = get_default_atol()
        supported_equalities = {
            Constraint.EQUAL_TO_ZERO,
            Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        }
        for constraint_id, constraint in self._ommx_instance.active_constraints.of_kind(
            ConstraintKind.Regular
        ).items():
            if constraint.equality not in supported_equalities:
                raise AssertionError(
                    "Unsupported constraint equality reached after applicability "
                    f"validation: {constraint.equality} for constraint "
                    f"{constraint_id}. This may indicate an OMMX implementation "
                    "bug; please report it to OMMX."
                )

            constraint_degree = constraint.function.degree()
            if constraint_degree is None:
                raise AssertionError(
                    "Non-polynomial constraint reached after applicability validation: "
                    f"constraint {constraint_id}. This may indicate an OMMX "
                    "implementation bug; please report it to OMMX."
                )

            if constraint_degree == 0 and not constraint.evaluate(
                {}, atol=atol
            ).is_feasible(atol=atol):
                raise OMMXDA4AdapterError(
                    f"Infeasible constant constraint was found: id {constraint_id}"
                )

    def _generate_binary_polynomial(self) -> BinaryPolynomial:
        """Generate BinaryPolynomial from OMMX instance.

        :return: BinaryPolynomial
        """
        instance = self._ommx_instance

        objective = instance.objective
        if objective.degree() is None:
            raise AssertionError(
                "Non-polynomial objective reached after applicability validation. "
                "This may indicate an OMMX implementation bug; please report it to "
                "OMMX."
            )

        if instance.sense not in (Sense.Minimize, Sense.Maximize):
            raise AssertionError(
                "Unsupported objective sense reached after applicability validation: "
                f"{instance.sense}. This may indicate an OMMX implementation bug; "
                "please report it to OMMX."
            )

        negate = instance.sense == Sense.Maximize
        binary_polynomial_terms = [
            BinaryPolynomialTerm(
                c=-value if negate else value,
                p=self._replace_polynomials_with_variable_map(key),
            )
            for key, value in objective.terms.items()
        ]

        # DA4 starts one-way one-hot groups at the minimum polynomial variable
        # and requires it in BinaryPolynomial even with coefficient 0. This
        # adapter maps native one-hot variables from 0, which the objective may omit.
        # Add a zero term only when the mapped variable 0 is absent from the objective.
        # Example: one-hot on x0, x1, x2 (numbers=[3]), objective x3 + 2*x4
        #   without this term: terms {p=[3], p=[4]}        -> group {3, 4, 5}
        #   with this term:    terms {p=[0], p=[3], p=[4]} -> group {0, 1, 2}
        if self._one_hot_dict and not any(
            0 in term.p for term in binary_polynomial_terms
        ):
            binary_polynomial_terms.append(BinaryPolynomialTerm(c=0.0, p=[0]))

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

        def add_squared_terms(function: Function) -> None:
            squared_function = function * function
            for key, value in squared_function.terms.items():
                add_term(key, value)

        for constraint in instance.active_constraints.of_kind(
            ConstraintKind.Regular
        ).values():
            # skip if not equality constraints
            if constraint.equality != Constraint.EQUAL_TO_ZERO:
                continue

            # Feasible constant constraints were validated before conversion.
            function = constraint.function
            if function.degree() == 0:
                continue

            add_squared_terms(function)

        # DA4 one-way one-hot groups cannot share decision variables. Treat each
        # group not passed to one_way_one_hot_groups as the regular equality
        # sum(x_i) - 1 = 0.
        for variables in self._penalty_one_hot_dict.values():
            add_squared_terms(
                Function(
                    Linear(
                        terms={variable: 1.0 for variable in variables},
                        constant=-1.0,
                    )
                )
            )

        # Check the final degree after squaring, binary simplification, and aggregation.
        # Squaring changes the scale. Even a small nonzero coefficient can
        # distinguish a feasible assignment from an infeasible one.
        penalty_degree = max(
            (len(key) for key, value in squared_terms_dict.items() if value != 0),
            default=0,
        )
        if penalty_degree > 2:
            raise OMMXDA4AdapterError(
                f"Penalty polynomial degree {penalty_degree} exceeds DA4's "
                "maximum supported degree of 2 after squaring, binary "
                "simplification (e.g., x² = x), and aggregation. "
                "Reformulate the equality constraints so that their squared penalties "
                "have degree at most 2 after binary simplification."
            )

        penalty_binary_polynomial_terms = [
            BinaryPolynomialTerm(
                c=value, p=self._replace_polynomials_with_variable_map(key)
            )
            for key, value in squared_terms_dict.items()
            if value != 0
        ]

        if len(penalty_binary_polynomial_terms) == 0:
            return None
        else:
            return PenaltyBinaryPolynomial(terms=penalty_binary_polynomial_terms)

    def _generate_inequalities(
        self, variables: list[DecisionVariable]
    ) -> list[Inequalities] | None:
        """Generate Inequalities from OMMX instance.

        :return: Inequalities
        """
        instance = self._ommx_instance

        inequalities_list = []
        for constraint_id, constraint in instance.active_constraints.of_kind(
            ConstraintKind.Regular
        ).items():
            # skip if not inequality constraints
            if constraint.equality != Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                continue

            # Feasible constant constraints were validated before conversion.
            function = constraint.function
            if function.degree() == 0:
                continue

            inequalities_terms = [
                BinaryPolynomialTerm(
                    c=value, p=self._replace_polynomials_with_variable_map(key)
                )
                for key, value in function.terms.items()
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

        # A singleton binary domain is a constraint on the search. DA4 can
        # override fixed_config, so encode x <= 0 or 1 - x <= 0 explicitly.
        for variable in variables:
            bound = variable.bound
            if bound.lower != bound.upper:
                continue
            index = self._variable_map[variable.id]
            if bound.lower == 0:
                terms = [BinaryPolynomialTerm(c=1, p=[index])]
            else:
                terms = [
                    BinaryPolynomialTerm(c=1, p=[]),
                    BinaryPolynomialTerm(c=-1, p=[index]),
                ]
            inequalities_list.append(Inequalities(terms=terms))

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

    def _generate_variable_map(
        self, variables: list[DecisionVariable]
    ) -> dict[int, int]:
        """Generate variable map that represents the correspondence
        between the IDs of decision variables in ommx.Instance and the variable numbers on QuboRequest.
        """
        variable_map = {}

        # First enumerate the decision variables from one-hot constraints,
        # then enumerate the remaining decision variables afterwards.
        index = 0

        for group in self._one_hot_dict.values():
            for variable in group:
                variable_map[variable] = index
                index += 1
        for decision_variable in variables:
            if decision_variable.kind != DecisionVariable.BINARY:
                raise AssertionError(
                    "Unsupported decision variable kind reached after applicability "
                    f"validation: {decision_variable.kind}. This may indicate an "
                    "OMMX implementation bug; please report it to OMMX."
                )

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
    ) -> tuple[
        dict[OneHotConstraintID, list[int]], dict[OneHotConstraintID, list[int]]
    ]:
        """Partition one-hot constraints into one_way_one_hot_groups and penalty handling.

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
        Prioritize longer constraints for one_way_one_hot_groups and send shorter
        ones to penalty handling
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
            instance.active_constraints.of_kind(ConstraintKind.OneHot).items(),
            key=lambda item: len(item[1].variables),
            reverse=True,
        )

        one_hot_dict: dict[OneHotConstraintID, list[int]] = {}
        penalty_one_hot_dict: dict[OneHotConstraintID, list[int]] = {}
        used_variables: set[int] = set()
        for constraint_id, one_hot_constraint in sorted_one_hot_constraints:
            variables = list(one_hot_constraint.variables)

            if used_variables.intersection(variables):
                penalty_one_hot_dict[constraint_id] = variables
                continue

            used_variables.update(variables)
            one_hot_dict[constraint_id] = variables

        return one_hot_dict, penalty_one_hot_dict
