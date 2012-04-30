#ifndef HOST_SOR_H
#define HOST_SOR_H

#include <cstdlib>
#include <math.h>
#include <iostream>

#define HOST_MAX_ITERATIONS 10000
#define HOST_EPS            .0000001
#define USE_APRIORI_DENOMS  1

void red_black_sor_double(double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
						  double* o, double* p, double* q,
						  double a, double b, double omegarelax, double residual_reduction, double dx,
						  double* h_p1, int nx, int ny, int nz);

void red_black_sor_float(float* e, float* f, float* g, float* h, float* m, float* n, float* r,
						 float* o, float* p, float* q,
						 float a, float b, float omegarelax, float residual_reduction, float dx,
						 float* h_p1, int nx, int ny, int nz);

void basic_sor_double(double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
					  double* o, double* p, double* q,
					  double a, double b, double omegarelax, double residual_reduction, double dx,
					  double* h_p1, int nx, int ny, int nz);

void basic_sor_float(float* e, float* f, float* g, float* h, float* m, float* n, float* r,
					 float* o, float* p, float* q,
					 float a, float b, float omegarelax, float residual_reduction, float dx,
					 float* h_p1, int nx, int ny, int nz);

void red_black_iter(double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
					double* o, double* p, double* q,
					double a, double b, double omegarelax, double residual_reduction, double dx,
					double* h_p1, int nx, int ny, int nz, int pass);

void red_black_iter(float* e, float* f, float* g, float* h, float* m, float* n, float* r,
					float* o, float* p, float* q,
					float a, float b, float omegarelax, float residual_reduction, float dx,
					float* h_p1, int nx, int ny, int nz, int pass);

void basic_iter(double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
				double* o, double* p, double* q,
				double a, double b, double omegarelax, double residual_reduction, double dx,
				double* h_p1, int nx, int ny, int nz);

void basic_iter(float* e, float* f, float* g, float* h, float* m, float* n, float* r,
				float* o, float* p, float* q,
				float a, float b, float omegarelax, float residual_reduction, float dx,
				float* h_p1, int nx, int ny, int nz);

void apriori_denoms(double* e, double* f, double* g, double* h, double* m, double* n,
					 double* r, double* o, double* p, double* q, double a, double b,
					 double omegarelax, double dx, int nx, int ny, int nz);

void apriori_denoms(float* e, float* f, float* g, float* h, float* m, float* n, 
					float* r, float* o, float* p, float* q, float a, float b, 
					float omegarelax, float dx, int nx, int ny, int nz);

#endif
