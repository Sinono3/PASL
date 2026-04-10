### Requirements

- uv
- Python >=3.8

### Installation

In a bash shell:

```
git clone https://github.com/AvLab-CV/PASL
cd PASL
git submodule init
git submodule update --recursive
uv venv
source .venv/bin/activate
uv pip install pip setuptools wheel torch==1.12.1
uv sync --no-build-isolation-package chumpy --no-build-isolation-package pytorch3d
```

### Requirements

For inference/eval:

- Extract contents of `DECA_data.zip` into `external/deca/data`

