# Installation

Python 3.12 or newer.

=== "uv"

    ```bash
    git clone https://github.com/GeomaticsCaminosUPM/ml_structural_system.git
    cd ml_structural_system
    uv sync
    ```

=== "pip"

    ```bash
    pip install git+https://github.com/GeomaticsCaminosUPM/ml_structural_system.git
    ```

    `requirements.txt` in the repository lists the tested versions: `pip install -r requirements.txt`.

This installs the `ml_structural_system` package and the `mlss` command. It also installs [footprint_attributes](https://github.com/GeomaticsCaminosUPM/footprint_attributes) from GitHub at a fixed commit, which computes the footprint features.

## GPU

XGBoost and CatBoost train on the GPU when a CUDA device is found, and on the CPU otherwise. Their standard wheels already include GPU support on Linux and Windows, so no extra installation is needed. Check what the package sees with:

```python
from ml_structural_system.gpu import cuda_available
cuda_available()
```

CatBoost reserves 95% of the free GPU memory by default. Set the environment variable `CATBOOST_GPU_RAM_PART` (for example `0.3`) to leave memory for other programs. It does not change the results.

## Development

```bash
uv sync --all-groups   # adds pytest and the documentation tools
uv run pytest          # tests
uv run mkdocs serve    # this documentation, at http://127.0.0.1:8000
```
