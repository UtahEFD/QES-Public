ADD FUTURE TESTS HERE

# COMPARING SOLVERS

This test allows the user to analyze the correlation in results between solvers.

## Usage -- 

Specify the options you want using CCmake, then rebuild and use ctest --verbose. When specifying options,
keep in mind that tests stack, so only run the tests you want to run, unless you want to wait a very long time. For
example, choosing ENABLE_ALL_COMPARISON_TESTS would be much faster than enabling each of the other tests individually,
since for each test it has to run the CPU solver again. ENABLE_ALL_COMPARISON_TESTS and ENABLE_ALL_TESTS will override any individually selected tests to reduce the amount of time spent.

## Options -- 

ENABLE_ALL_COMPARISON_TESTS: Runs every solver. 
ENABLE_DYNAMIOC_PARALLELISM_COMPARISON_TESTS: Only runs the CPU and Dynamic Parallel solvers.
ENABLE_GLOBAL_COMPARISON_TESTS: Only runs the CPU and Global Memory solvers.
ENABLE_SHARED_COMPARISON_TESTS: Only runs the CPU and Shared Memory solvers.
ENABLE_LONG_COMPARISON_TESTS: By default, only FlatTerrain is used for the test, which takes around 20-30 seconds. Turning
this on adds GaussianHill to the test, which adds around 5 minutes.
ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS: Adds AskerveinHill to the test, which adds around 15 minutes.

## Running compareSolvers manually --

Runs using similar arguments to qesWinds. From the build folder, run: ./tests/compareSolvers -q "..." -s "...".
The -q command specifies which input file you want to use (example: "../data/InputFiles/FlatTerrain.xml".
The -s command now specifies which comparison to run. 1 compares all solvers, 2 compares the Dynamic Parallel solver and
CPU solvers, 3 compares the Global Memory and CPU solvers, and 4 compares the Shared Memory and CPU solvers.

# QES regeression tests suites.


## Run tests 
By default ENABLE_GPU_TESTS is true. Test are run using:
```
make test
```

Other tests are available: ENABLE_CPU_TESTS, ENABLE_LONG_SANITY_TESTS, ENABLE_ALL_TESTS. Tests can be turned on using:
```
cmake -DXXX=true ..
```
where XXX is the options above.

List of test: 
- short tests: FlatTerrain, GaussianHill, OklahomaCity)
- long tests: SaltLakeCity, RxCADRE

## Regression tests
