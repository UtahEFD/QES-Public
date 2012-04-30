/**
*	Author: Pete Willemsen <willemsn@d.umn.edu>
* Reason: Collect the C++/Fortran interface into one file that handles multiple
*         platforms.
*/

#ifndef __INC_FORTRAN_DATAMODULE_H__
#define __INC_FORTRAN_DATAMODULE_H__ 1

#if defined(__APPLE__) || (__GNUC__ == 4 && __GNUC_MINOR__ > 2)
#define F90DATAMODULE_(v) __datamodule_MOD_##v
#else
#define F90DATAMODULE_(v) __datamodule__##v
#endif

extern "C" int F90DATAMODULE_(roofflag);

extern "C" int F90DATAMODULE_(soriter);

extern "C" double* F90DATAMODULE_(e);
extern "C" double* F90DATAMODULE_(f);
extern "C" double* F90DATAMODULE_(g);
extern "C" double* F90DATAMODULE_(h);

extern "C" double* F90DATAMODULE_(m);
extern "C" double* F90DATAMODULE_(n);
extern "C" double* F90DATAMODULE_(r);
extern "C" double* F90DATAMODULE_(o);

extern "C" double* F90DATAMODULE_(p);
extern "C" double* F90DATAMODULE_(q);
extern "C" double* F90DATAMODULE_(p1);
extern "C" double* F90DATAMODULE_(p2);

extern "C" double* F90DATAMODULE_(x);
extern "C" double* F90DATAMODULE_(y);
extern "C" double* F90DATAMODULE_(z);
extern "C" double* F90DATAMODULE_(uo);
extern "C" double* F90DATAMODULE_(vo);
extern "C" double* F90DATAMODULE_(wo);
extern "C" double* F90DATAMODULE_(u);
extern "C" double* F90DATAMODULE_(v);
extern "C" double* F90DATAMODULE_(w);
extern "C" double* F90DATAMODULE_(visc);
extern "C" double* F90DATAMODULE_(pwtx);
extern "C" int* F90DATAMODULE_(icellflag);
extern "C" int F90DATAMODULE_(nx);
extern "C" int F90DATAMODULE_(ny);
extern "C" int F90DATAMODULE_(nz);
extern "C" int F90DATAMODULE_(num_time_steps);
extern "C" int F90DATAMODULE_(sor_iter);
extern "C" int F90DATAMODULE_(itermax);
extern "C" int F90DATAMODULE_(i_time);
extern "C" double F90DATAMODULE_(time);
extern "C" double F90DATAMODULE_(time_incr);
//extern "C" double F90DATAMODULE_(start_time);

extern "C" double F90DATAMODULE_(a);
extern "C" double F90DATAMODULE_(b);
extern "C" double F90DATAMODULE_(omegarelax);
extern "C" double F90DATAMODULE_(residual_reduction);
extern "C" double F90DATAMODULE_(abse);
extern "C" double F90DATAMODULE_(eps);

// used for euler
extern "C" double F90DATAMODULE_(dx), F90DATAMODULE_(dy), F90DATAMODULE_(dz);
extern "C" double F90DATAMODULE_(alpha1), F90DATAMODULE_(alpha2);

extern "C"
{
  void init_();
  void sort_();
  void building_parameterizations_();
  void sensorinit_();
  void divergence_();
  void denominators_();
  void sor3d_();
  void euler_();
  void diffusion_();
  void outfile_();
}

#endif //  __INC_FORTRAN_DATAMODULE_H__ 1
