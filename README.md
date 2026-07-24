<div align="center">

# QES: Quick Environmental Simulations

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7098279.svg)](https://doi.org/10.5281/zenodo.7098279)

</div>

The Quick Environmental Simulation (**QES**) code is a low-computational-cost framework designed to compute high-resolution wind and concentration fields in complex atmospheric-boundary-layer environments. QES is written in C++ and (optionally) NVIDIA CUDA for GPU acceleration.

**Python bindings** live in a separate repository: [rupeelab17/pyQES](https://github.com/rupeelab17/pyQES) (`pip install pyqes`). That repo vendors this tree as the `qes-core` git submodule.

> GPU acceleration requires an NVIDIA GPU with Compute Capability 7.0+. The code can be compiled and run on CPU without CUDA.

---

## Table of contents

- [QES modules](#qes-modules)
- [C++ package requirements](#package-requirements)
- [Building the C++ code](#building-the-code)
- [Running QES (CLI)](#running-qes)
- [Testing](#testing)
- [Documentation](#building-the-documentation-via-doxygen)
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

### Library-only build (for pyQES)

When this tree is used as the `qes-core` submodule of [pyQES](https://github.com/rupeelab17/pyQES), configure with:

```bash
cmake -DQES_BUILD_APPS=OFF ..
```

This builds only the core libraries (`qesutil`, `qeswindscore`, …) with position-independent code — no CLI executables, examples, or CPack packages.

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

Wheel / Python packaging uses vcpkg from the [pyQES](https://github.com/rupeelab17/pyQES) repository. For a native CLI build you can still use [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get-started/overview) and [`CMakePresets.json`](CMakePresets.json) with a local toolchain:

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

---

## Building the Documentation via Doxygen

After configuring the C++ build:

```bash
make windsdoc
```

Output updates the `html` and `latex` folders under `docs/`. Online docs: [qes-documentation.readthedocs.io](https://qes-documentation.readthedocs.io/en/latest).

---

## Published QES Papers

1. B. Bozorgmehr et al., “Utilizing dynamic parallelism in CUDA to accelerate a 3D red-black successive over relaxation wind-field solver,” *Environ Modell Softw*, vol. 137, p. 104958, 2021, doi: [10.1016/j.envsoft.2021.104958](https://doi.org/10.1016/j.envsoft.2021.104958).

2. F. Margairaz et al., “Development and evaluation of an isolated-tree flow model for neutral-stability conditions,” *Urban Clim*, vol. 42, p. 101083, 2022, doi: [10.1016/j.uclim.2022.101083](https://doi.org/10.1016/j.uclim.2022.101083).

3. M. J. Moody et al., “QES-Fire: a dynamically coupled fast-response wildfire model,” *Int J Wildland Fire*, vol. 31, no. 3, pp. 306–325, 2022, doi: [10.1071/wf21057](https://doi.org/10.1071/WF21057).
