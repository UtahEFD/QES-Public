<div align="center">

# QES: Quick Environmental Simulations

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7098279.svg)](https://doi.org/10.5281/zenodo.7098279)

</div>

The Quick Environmental Simulation (**QES**) code is a low-computational-cost framework designed to compute high-resolution wind and concentration fields in complex atmospheric-boundary-layer environments. QES is written in C++ and (optionally) NVIDIA CUDA for GPU acceleration.

This repository also ships **pyQES**, a Python package that wraps QES-Winds, QES-Plume and QES-Fire via pybind11, with Pydantic configuration models and geospatial preprocessing helpers.

> GPU acceleration requires an NVIDIA GPU with Compute Capability 7.0+. The code can be compiled and run on CPU without CUDA.

---

## Table of contents

- [QES modules](#qes-modules)
- [pyQES (Python)](#pyqes-python)
- [C++ package requirements](#package-requirements)
- [Building the C++ code](#building-the-code)
- [Running QES (CLI)](#running-qes)
- [Testing](#testing)
- [Documentation](#building-the-documentation-via-doxygen)
- [Continuous Integration](#continuous-integration)
- [Published papers](#published-qes-papers)

---

## QES modules

### QES-Winds

QES-Winds is a fast-response 3D diagnostic urban wind model using a mass-conserving wind-field solver. It uses a variational analysis technique to ensure mass conservation, solving a Poisson equation for Lagrange multipliers with the Successive Over-Relaxation (SOR) method.

> B. Bozorgmehr et al., “Utilizing dynamic parallelism in CUDA to accelerate a 3D red-black successive over relaxation wind-field solver,” *Environ Modell Softw*, vol. 137, p. 104958, 2021, doi: [10.1016/j.envsoft.2021.104958](https://doi.org/10.1016/j.envsoft.2021.104958).

### QES-Turb

QES-Turb is a turbulence model based on Prandtl’s mixing-length and Boussinesq eddy-viscosity hypotheses. It computes the stress tensor using local velocity gradients and empirical non-local parameterizations.

### QES-Plume

QES-Plume is a stochastic Lagrangian dispersion model using QES-Winds mean wind fields and QES-Turb turbulence fields. It solves the generalized Langevin equations and can also run stand-alone with fields from RANS or LES models.

> F. Margairaz et al., "QES-Plume v1.0: A Lagrangian dispersion model," *Geosci Model Dev* (submitted).

### QES-Fire

QES-Fire is a microscale wildfire model coupling the fire front to microscale winds (rate of spread, kinematic plume-rise, mass-consistent wind solver).

> M. J. Moody et al., “QES-Fire: a dynamically coupled fast-response wildfire model,” *Int J Wildland Fire*, vol. 31, no. 3, pp. 306–325, 2022, doi: [10.1071/wf21057](https://doi.org/10.1071/WF21057).

---

## pyQES (Python)

**pyQES** exposes the QES solvers as a Python package:

| Submodule | Role |
|-----------|------|
| `pyQES.pywinds` | Run QES-Winds (`run(...)`) |
| `pyQES.pyplume` | Run QES-Plume |
| `pyQES.pyfire` | Run coupled QES-Fire |
| `pyQES.util` | Pydantic config, XML/JSON I/O, geospatial helpers |

Requires **Python ≥ 3.10**. Native dependencies for the extension build: **Boost**, **NetCDF-C++**, **GDAL** (and a C++17 compiler). CUDA is optional.

### Install (development)

From the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
# Native libs (macOS Homebrew example)
brew install boost netcdf-cxx gdal

# Editable install + extension build
uv sync
```

On Linux, install the equivalent packages (`libboost-dev`, `libnetcdf-c++4-dev`, `libgdal-dev`, …) or use [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/overview) as for the C++ build.

Optional extras (also pulled by the `dev` dependency group):

```bash
uv sync --extra geo --extra io   # rasterio/pyproj/geopandas + netCDF4
```

Wheel builds for macOS / Linux / Windows are produced by GitHub Actions (`wheels.yml`) and published to PyPI on `v*` tags (`publish.yml`).

> **macOS tip:** if `uv sync` builds an `x86_64` wheel on Apple Silicon, check that `ARCHFLAGS` is not forced to `-arch x86_64` in your shell profile. Prefer `export ARCHFLAGS="-arch $(uname -m)"`.

### Quick start

```python
from pyQES import pywinds
from pyQES.util.config import WindsParameters, SensorParameters, TimeSeries

# From an existing QES XML
result = pywinds.run(
    xml="data/umep_workflow/qes/umep_larochelle.xml",
    dem="data/umep_workflow/DEM_clip.tif",
    buildings_src="data/umep_workflow/buildings.shp",
    buildings_mask="data/umep_workflow/mask.shp",
    solver="cpu",
    out_basename="umep_larochelle",
    work_dir="data/umep_workflow/output",
    auto_preprocess=True,
)
print(result.winds_out)

# Or fully in Python (no XML file)
params = WindsParameters()
params.simulation_parameters.dem = "data/umep_workflow/DEM_clip.tif"
params.simulation_parameters.cell_size = (2.0, 2.0, 0.5)
params.simulation_parameters.halo_x = 40.0
params.simulation_parameters.halo_y = 40.0
params.simulation_parameters.domain_rotation = 0.0  # must be 0

sensor = SensorParameters(
    time_series=[TimeSeries(speed=3.0, direction=270.0, height=10.0, site_z0=0.24)]
)
result = pywinds.run(config=params, sensor=sensor, solver="cpu", work_dir="/tmp/qes_out")
```

`pywinds.run` accepts **XML**, **JSON**, a **`WindsParameters`** object, and/or keyword overrides. With `auto_preprocess=True` it computes DEM origin / domain cell counts, places the sensor at the DEM north center, and optionally clips buildings (`buildings_src` + `buildings_mask`).

### Example scripts (La Rochelle / UMEP)

Under [`data/umep_workflow/`](data/umep_workflow/):

| Script | Description |
|--------|-------------|
| [`run_qeswinds.py`](data/umep_workflow/run_qeswinds.py) | Run from `qes/umep_larochelle.xml` (mirror of the bash wrappers) |
| [`run_qeswinds_args.py`](data/umep_workflow/run_qeswinds_args.py) | Same case with **all** parameters as CLI args (no XML read) |
| [`run_qeswinds.sh`](data/umep_workflow/run_qeswinds.sh) / `_cpu` / `_gpu` | Original bash launchers (C++ binary) |

```bash
# XML-based
uv run python data/umep_workflow/run_qeswinds.py
uv run python data/umep_workflow/run_qeswinds.py --no-preprocess

# Fully argument-driven (defaults = La Rochelle case)
uv run python data/umep_workflow/run_qeswinds_args.py
uv run python data/umep_workflow/run_qeswinds_args.py --speed 5 --direction 180
```

### Python tests

```bash
# Fast unit tests (config, XML I/O, geo) — no full solver run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/python -m "not slow"

# End-to-end winds run (requires compiled extension + geo deps)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/python -m slow
```

---

## Package Requirements

**QES requires C++17.**

**GPU builds** need the CUDA toolkit and an NVIDIA GPU with Compute Capability 7.0 or higher. CPU-only builds are supported.

On a general Linux system (e.g. Ubuntu), install:

* `libgdal-dev`
* `libnetcdf-c++4-dev`
* `libnetcdf-cxx-legacy-dev`
* `libnetcdf-dev`
* `netcdf-bin`
* `libboost-all-dev`
* `cmake`
* `cmake-curses-gui`

With `apt`:

```bash
apt install libgdal-dev libnetcdf-c++4-dev libnetcdf-cxx-legacy-dev \
  libnetcdf-dev netcdf-bin libboost-all-dev cmake cmake-curses-gui
```

CUDA has been tested with 11.8. Optionally, NVIDIA OptiX (tested up to 7.5 / 7.6) can accelerate mixing-length calculations.

On **macOS** (Homebrew):

```bash
brew install boost netcdf-cxx gdal cmake
```

---

## Building the Code

On the public repository, the most recent released version is on the `main` branch.

### Building on a general Linux / macOS system

```bash
mkdir build && cd build
cmake ..
make
```

### Build types

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Supported types: `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`. Use **Release** for production.

### Python extension build (CMake)

When building via `uv sync` / scikit-build-core, CMake is configured with `QES_BUILD_PYTHON=ON`, which builds the pybind11 modules under `src/bindings` and installs them into the `pyQES` package (executables / C++ tests are skipped).

### Building on CHPC (University of Utah)

Preferred setup: GCC 11.2 + CUDA 11.8.

```bash
source CHPC/loadmodules_QES.sh
# or load modules manually — see CHPC/loadmodules_QES.sh

mkdir build && cd build
cmake -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
make
```

For OptiX support, add:

```bash
-DOptiX_INSTALL_DIR=/uufs/chpc.utah.edu/sys/installdir/optix/7.6.0
```

### vcpkg — Windows, macOS and Linux

QES supports [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/overview) via [`vcpkg.json`](vcpkg.json) and [`CMakePresets.json`](CMakePresets.json).

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh   # or bootstrap-vcpkg.bat on Windows
export VCPKG_ROOT=/path/to/vcpkg
```

Then from the QES source tree:

```bash
cmake --preset=macOSDev      # or windowsDev, linuxDev, …
cmake --build --preset=macOSDev
```

**Windows:** use Visual Studio Community (MSVC). Open the repo folder, select the `windowsDev` configuration, then Build All. Executables land in the preset build directory.

---

## Running QES

### Command line (C++ binaries)

```bash
./qesWinds/qesWinds -q ../data/InputFiles/GaussianHill.xml -s 2 -w -o gaussianHill
./qesWinds/qesWinds -?    # help
```

Solver type (`-s`): `1` = CPU, `2` = GPU (dynamic parallelism).

### umep_workflow (bash)

```bash
./data/umep_workflow/run_qeswinds_cpu.sh
./data/umep_workflow/run_qeswinds_gpu.sh
```

### Slurm template (CHPC)

```bash
#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=qesGaussian
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.5.0
ulimit -c unlimited -s
./qesWinds/qesWinds -q ../data/InputFiles/GaussianHill.xml -s 2 -w -o gaussianHill
```

---

## Testing

### C++ (`ctest`)

```bash
ctest                 # all tests
ctest --verbose
ctest -N              # list
ctest -R $testname
```

Enable sanity / GPU tests:

```bash
cmake -DENABLE_SANITY_TESTS=ON -DENABLE_GPU_TESTS=ON ..
```

QES-Winds sanity tests include: `GPU_FlatTerrain`, `GPU_GaussianHill`, `GPU_OklahomaCity`, `GPU_MultiSensors`, `GPU_SaltLakeCity`, `GPU_RxCADRE`.

Unit tests:

```bash
cmake -DENABLE_UNITTESTS=ON ..
```

### Python

See [Python tests](#python-tests) above (`tests/python/`).

---

## Building the Documentation via Doxygen

After configuring the C++ build:

```bash
make windsdoc
```

Output updates the `html` and `latex` folders under `docs/`. Online docs: [qes-documentation.readthedocs.io](https://qes-documentation.readthedocs.io/en/latest).

---

## Continuous Integration

GitHub Actions workflows:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | push / PR (Python paths) | Ruff, mypy, fast pytest |
| [`.github/workflows/wheels.yml`](.github/workflows/wheels.yml) | PR / dispatch / reusable | Cross-platform wheels (cibuildwheel + vcpkg) |
| [`.github/workflows/publish.yml`](.github/workflows/publish.yml) | tag `v*` | Build wheels and publish to PyPI (Trusted Publishing) |

The project [`README.md`](README.md) is declared as the package long description in [`pyproject.toml`](pyproject.toml) (`project.readme`). It is embedded in the wheel/sdist `METADATA` / `PKG-INFO` (Markdown) and therefore shown on the PyPI project page when a release is published.

---

## Published QES Papers

1. B. Bozorgmehr et al., “Utilizing dynamic parallelism in CUDA to accelerate a 3D red-black successive over relaxation wind-field solver,” *Environ Modell Softw*, vol. 137, p. 104958, 2021, doi: [10.1016/j.envsoft.2021.104958](https://doi.org/10.1016/j.envsoft.2021.104958).

2. F. Margairaz et al., “Development and evaluation of an isolated-tree flow model for neutral-stability conditions,” *Urban Clim*, vol. 42, p. 101083, 2022, doi: [10.1016/j.uclim.2022.101083](https://doi.org/10.1016/j.uclim.2022.101083).

3. M. J. Moody et al., “QES-Fire: a dynamically coupled fast-response wildfire model,” *Int J Wildland Fire*, vol. 31, no. 3, pp. 306–325, 2022, doi: [10.1071/wf21057](https://doi.org/10.1071/WF21057).
