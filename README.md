# ommx-da4-adapter
This package provides an adapter for [Fujitsu Digital Annealer(DA4)](https://www.fujitsu.com/jp/digitalannealer/) from [OMMX](https://github.com/Jij-Inc/ommx). It allows you to solve optimization problems defined in OMMX format using DA4's powerful solver.

## Installation
The `ommx-da4-adapter` can be installed from PyPI as follows:

```bash
pip install ommx-da4-adapter
```

## Usage
Here's a simple example of how to use the adapter directly:

```python
from ommx.v1 import Instance, DecisionVariable
from ommx_da4_adapter import OMMXDA4Adapter

x_0 = DecisionVariable.binary(id=0, name="x_0")
x_1 = DecisionVariable.binary(id=1, name="x_1")

ommx_instance = Instance.from_components(
    decision_variables=[x_0, x_1],
    objective=x_0 * x_1 + x_0 - x_1 + 1,
    constraints=[x_0 + x_1 == 1],
    sense=Instance.MINIMIZE,
)

ommx_sampleset = OMMXDA4Adapter.sample(
    ommx_instance=ommx_instance,
    token="*** your da4 api token ***",
    url="*** da4 url ***",
)
```

You can also use the adapter and client separately:

```python
from ommx_da4_adapter import OMMXDA4Adapter, DA4Client

# Assuming ommx_instance is already defined as above
adapter = OMMXDA4Adapter(ommx_instance)

qubo_request = adapter.sampler_input

client = DA4Client(
    token="*** your da4 api token ***",
    url="*** da4 url ***",
)

qubo_response = client.sample(qubo_request=qubo_request)

ommx_sampleset = adapter.decode_to_sampleset(qubo_response)
```

