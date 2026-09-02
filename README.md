# ommx-da4-adapter
This package provides an adapter for [Fujitsu Digital Annealer(DA4)](https://www.fujitsu.com/jp/digitalannealer/) from [OMMX](https://github.com/Jij-Inc/ommx). It allows you to solve optimization problems defined in OMMX format using DA4's powerful solver.

## Installation
The `ommx-da4-adapter` can be installed from PyPI as follows:

```bash
pip install ommx-da4-adapter
```

## Usage

Here's a simple example using `sample()`:

```python
from ommx import Instance
from ommx_da4_adapter import OMMXDA4Adapter

ommx_instance = Instance.minimize()
x_0 = ommx_instance.new_binary("x_0")
x_1 = ommx_instance.new_binary("x_1")
ommx_instance.objective = x_0 * x_1 + x_0 - x_1 + 1
ommx_instance.add_constraint(x_0 + x_1 == 1)

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

You can use it with
`DA4Client` directly when you need access to the request and response:

```python
from ommx_da4_adapter import OMMXDA4Adapter, DA4Client

adapter = OMMXDA4Adapter(ommx_instance)

qubo_request = adapter.sampler_input

client = DA4Client(
    token="*** your da4 api token ***",
    url="*** da4 url ***",
)

qubo_response = client.sample(qubo_request=qubo_request)

ommx_sampleset = adapter.decode_to_sampleset(qubo_response)
```
