project(Urbfortran Fortran)

add_library (Urbfortran building_parameterizations.f90 bisect.f90 building_connect.f90 	datamodule.f90 defbuild.f90 divergence.f90 euler.f90 init.f90 main.f90 outfile.f90 pentagon.f90 plantinit.f90 regress.f90 sensorinit.f90 sor3d.f90 sort.f90 street_intersect.f90 upwind.f90 zd_bisect.f90 diffusion.f90 turbulence_model.f90 poisson.f90 wallbc.f90 read_quic_met.f90 read_ittmm5_met.f90 read_hotmac_met.f90 cylinderwake.f90 rectanglewake.f90 wake.f90 streetcanyon.f90 rooftop.f90 parking_garage.f90 build_garage.f90 unbuild_garage.f90 courtyard.f90 bridgewake.f90 denominators.f90)

# DO I NEED THIS???  ... I'm not specifying the defaults...
# FC = gfortran
# FFLAGS = -O2 -fdefault-double-8 -fdefault-real-8 -fdefault-integer-8 -ffree-form
# F95 = gfortran
# F95FLAGS = -O3 -fdefault-double-8 -fdefault-real-8 -fdefault-integer-8 -ffree-form
