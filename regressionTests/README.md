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
