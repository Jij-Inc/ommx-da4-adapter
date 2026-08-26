# ommx-da4-adapter
This package provides an adapter for [Fujitsu Digital Annealer(DA4)](https://www.fujitsu.com/jp/digitalannealer/) from [OMMX](https://github.com/Jij-Inc/ommx). It allows you to solve optimization problems defined in OMMX format using DA4's powerful solver.

## Installation
The `ommx-da4-adapter` can be installed from PyPI as follows:

```bash
pip install ommx-da4-adapter
```

## Usage

The easy `sample()` and `solve()` APIs prepare an isolated copy of the input
with the adapter's recommended policy. The caller's `Instance` is not modified.

Here's a simple example using `sample()`:

```python
from ommx import Instance, DecisionVariable
from ommx_da4_adapter import OMMXDA4Adapter

x_0 = DecisionVariable.binary(id=0, name="x_0")
x_1 = DecisionVariable.binary(id=1, name="x_1")

ommx_instance = Instance.from_components(
    decision_variables=[x_0, x_1],
    objective=x_0 * x_1 + x_0 - x_1 + 1,
    constraints={0: x_0 + x_1 == 1},
    sense=Instance.MINIMIZE,
)

ommx_sampleset = OMMXDA4Adapter.sample(
    ommx_instance=ommx_instance,
    token="*** your da4 api token ***",
    url="*** da4 url ***",
)
```

DA4 directly accepts binary polynomial minimization and maximization problems
with equality, less-than-or-equal, and OneHot constraints. The recommended
preparation policy preserves OneHot constraints for DA4's native one-way
one-hot groups and lowers unsupported Indicator and SOS1 constraints.

Use `sample_without_preparation()` or `solve_without_preparation()` when you
want to prepare the instance explicitly. These preparation-free APIs require
the input to belong to `INPUT_CLASS` and do not modify it:

```python
import copy

prepared_instance = copy.copy(ommx_instance)
prepared_instance.prepare(
    OMMXDA4Adapter.INPUT_CLASS,
    OMMXDA4Adapter.recommended_preparation_policy(),
)

ommx_sampleset = OMMXDA4Adapter.sample_without_preparation(
    prepared_instance,
    token="*** your da4 api token ***",
    url="*** da4 url ***",
)
```

The constructor also requires an exact `INPUT_CLASS` member. You can use it
with `DA4Client` directly when you need access to the request and response:

```python
from ommx_da4_adapter import OMMXDA4Adapter, DA4Client

adapter = OMMXDA4Adapter(prepared_instance)

qubo_request = adapter.sampler_input

client = DA4Client(
    token="*** your da4 api token ***",
    url="*** da4 url ***",
)

qubo_response = client.sample(qubo_request=qubo_request)

ommx_sampleset = adapter.decode_to_sampleset(qubo_response)
```
