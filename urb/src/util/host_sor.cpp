#include "host_sor.h"

void red_black_sor_double
(
	double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
  	double* o, double* p, double* q,
	double a, double b, double omegarelax, double residual_reduction, double dx,
	double* h_p1, int nx, int ny, int nz
) 
{
	bool converged = false;

	double* h_p2 = (double*) malloc(nx * ny * nz * sizeof(double));

	for(int i = 0; i < nx * ny * nz; i++) {h_p1[i] = 0.0;}
	for(int i = 0; i < nx * ny * nz; i++) {h_p2[i] = 0.0;}

	double abse = 0.0;
	double eps  = 1.0;
	double* err = (double*) malloc(nx * ny * nz * sizeof(double));

	int iterations = 0;

	while(iterations < HOST_MAX_ITERATIONS && !converged) 
	{
		red_black_iter(e, f, g, h, m, n, r, o, p, q, a, b, omegarelax, residual_reduction, dx,
				   		h_p1, nx, ny, nz, 0);
		red_black_iter(e, f, g, h, m, n, r, o, p, q, a, b, omegarelax, residual_reduction, dx,
				   		h_p1, nx, ny, nz, 1);

		for(int i = 0; i < nx * ny; i++) {h_p1[i] = h_p1[nx * ny + i];} //boundry
		for(int i = 0; i < nx * ny * nz; i++) 
		{
			abse += fabs(h_p1[i] - h_p2[i]); //error
			h_p2[i] = h_p1[i];
		}
		abse = abse / (nx * ny * nz);
		if(iterations == 0) {eps = abse / pow(10, residual_reduction);}
		if(abse < eps || abse < HOST_EPS) {converged = true;}
		
		iterations++;
	}

	std::cout << "Iterations: " << iterations << std::endl;
	std::cout << "Error Tolerance (eps): " << eps << std::endl;
	std::cout << "Omegarelax: " << omegarelax << std::endl;
	free(h_p2); free(err);
}

void red_black_sor_float
(
	float* e, float* f, float* g, float* h, float* m, float* n, float* r,
	float* o, float* p, float* q,
	float a, float b, float omegarelax, float residual_reduction, float dx,
	float* h_p1, int nx, int ny, int nz
) 
{
	bool converged = false;

	float* h_p2 = (float*) malloc(nx * ny * nz * sizeof(float));

	for(int i = 0; i < nx * ny * nz; i++) {h_p1[i] = 0.0;}
	for(int i = 0; i < nx * ny * nz; i++) {h_p2[i] = 0.0;}

	float abse = 0.0;
	float eps  = 1.0;
	float* err = (float*) malloc(nx * ny * nz * sizeof(float));

	int iterations = 0;

	while(iterations < HOST_MAX_ITERATIONS && !converged) 
	{
		red_black_iter(e, f, g, h, m, n, r, o, p, q, a, b, omegarelax, residual_reduction, dx,
				   		h_p1, nx, ny, nz, 0);
		red_black_iter(e, f, g, h, m, n, r, o, p, q, a, b, omegarelax, residual_reduction, dx,
				   		h_p1, nx, ny, nz, 1);

		for(int i = 0; i < nx * ny; i++) {h_p1[i] = h_p1[nx * ny + i];} //boundry
		for(int i = 0; i < nx * ny * nz; i++) 
		{
			abse += fabs(h_p1[i] - h_p2[i]); //error
			h_p2[i] = h_p1[i];
		}
		abse = abse / (nx * ny * nz);
		if(iterations == 0) {eps = abse / pow(10, residual_reduction);}
		if(abse < eps || abse < HOST_EPS) {converged = true;}
		
		iterations++;
	}

	std::cout << "Iterations: " << iterations << std::endl;
	std::cout << "Error Tolerance (eps): " << eps << std::endl;
	std::cout << "Omegarelax: " << omegarelax << std::endl;
	free(h_p2); free(err);
}

void basic_sor_double
(
	double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
	double* o, double* p, double* q,
	double a, double b, double omegarelax, double residual_reduction, double dx,
	double* h_p1, int nx, int ny, int nz
)
{
	bool converged = false;

	double* h_p2 = (double*) malloc(nx * ny * nz * sizeof(double));

	for(int i = 0; i < nx * ny * nz; i++) {h_p1[i] = 0.0;}
	for(int i = 0; i < nx * ny * nz; i++) {h_p2[i] = 0.0;}

	double abse = 0.0;
	double eps  = 1.0;
	double* err = (double*) malloc(nx * ny * nz * sizeof(double));

	int iterations = 0;

	while(iterations < HOST_MAX_ITERATIONS && !converged) 
	{
		basic_iter(e, f, g, h, m, n, r, o, p, q, a, b, omegarelax, residual_reduction, dx,
				   		h_p1, nx, ny, nz);

		for(int i = 0; i < nx * ny; i++) {h_p1[i] = h_p1[nx * ny + i];} //boundry
		for(int i = 0; i < nx * ny * nz; i++) {abse += fabs(h_p1[i] - h_p2[i]);} //error
		for(int i = 0; i < nx * ny * nz; i++) {h_p2[i] = h_p1[i];}
		abse = abse / (nx * ny * nz);
		if(iterations == 0) {eps = abse / pow(10, residual_reduction);}
		if(abse < eps || abse < HOST_EPS) {converged = true;}

		iterations++;
	}

	std::cout << "Iterations: " << iterations << std::endl;
	std::cout << "Error Tolerance (eps): " << eps << std::endl;
	std::cout << "Omegarelax: " << omegarelax << std::endl;
	free(h_p2); free(err);
}

void basic_sor_float
(
	float* e, float* f, float* g, float* h, float* m, float* n, float* r,
	float* o, float* p, float* q,
	float a, float b, float omegarelax, float residual_reduction, float dx,
	float* h_p1, int nx, int ny, int nz
)
{
	bool converged = false;

	float* h_p2 = (float*) malloc(nx * ny * nz * sizeof(float));

	for(int i = 0; i < nx * ny * nz; i++) {h_p1[i] = 0.0;}
	for(int i = 0; i < nx * ny * nz; i++) {h_p2[i] = 0.0;}

	float abse = 0.0;
	float eps  = 1.0;
	float* err = (float*) malloc(nx * ny * nz * sizeof(float));

	int iterations = 0;

	while(iterations < HOST_MAX_ITERATIONS && !converged) 
	{
		basic_iter(e, f, g, h, m, n, r, o, p, q, a, b, omegarelax, residual_reduction, dx,
				   		h_p1, nx, ny, nz);

		for(int i = 0; i < nx * ny; i++) {h_p1[i] = h_p1[nx * ny + i];} //boundry
		for(int i = 0; i < nx * ny * nz; i++) {abse += fabs(h_p1[i] - h_p2[i]);} //error
		for(int i = 0; i < nx * ny * nz; i++) {h_p2[i] = h_p1[i];}
		abse = abse / (nx * ny * nz);
		if(iterations == 0) {eps = abse / pow(10, residual_reduction);}
		if(abse < eps || abse < HOST_EPS) {converged = true;}

		iterations++;
	}

	std::cout << "Iterations: " << iterations << std::endl;
	std::cout << "Error Tolerance (eps): " << eps << std::endl;
	std::cout << "Omegarelax: " << omegarelax << std::endl;
	free(h_p2); free(err);
}


void red_black_iter
(
	double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
	double* o, double* p, double* q,
	double a, double b, double omegarelax, double residual_reduction, double dx,
	double* h_p1, int nx, int ny, int nz, int pass
)
{
	if(USE_APRIORI_DENOMS) 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			if(((pass + k + j + i) % 2) != pass) {return;}
			int cI = k * nx * ny + j * nx + i;
			h_p1[cI] = (
						 e[cI] * h_p1[cI + 1]       + f[cI] * h_p1[cI - 1]
					   + g[cI] * h_p1[cI + nx]      + h[cI] * h_p1[cI - nx]
					   + m[cI] * h_p1[cI + nx * ny] + n[cI] * h_p1[cI - nx * ny]
					   - r[cI]
					   )
					   +
					   (1 - omegarelax) * h_p1[cI];

		}
	}
	else {
		//Do the reds...
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			if(((pass + k + j + i) % 2) != pass) {return;}
			int cI = k * nx * ny + j * nx + i;
			h_p1[cI] = omegarelax * (
										(
											  e[cI] * h_p1[cI + 1]       +     f[cI] * h_p1[cI - 1]
										+ a * g[cI] * h_p1[cI + nx]      + a * h[cI] * h_p1[cI - nx]
										+ b * m[cI] * h_p1[cI + nx * ny] + b * n[cI] * h_p1[cI - nx * ny]
										- dx * dx * r[cI]
										)
										/
										(
										2.0 * (o[cI] + a * p[cI] + b * q[cI])
										)
									)
					   +
					   (1 - omegarelax) * h_p1[cI];
		}
	}
}

void red_black_iter
(
	float* e, float* f, float* g, float* h, float* m, float* n, float* r,
	float* o, float* p, float* q,
	float a, float b, float omegarelax, float residual_reduction, float dx,
	float* h_p1, int nx, int ny, int nz, int pass
)
{
	if(USE_APRIORI_DENOMS) 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) {
			if(((k + j + i) % 2) != pass) {}
			else 
			{
				int cI = k * nx * ny + j * nx + i;
				h_p1[cI] = (
							 e[cI] * h_p1[cI + 1]       + f[cI] * h_p1[cI - 1]
						   + g[cI] * h_p1[cI + nx]      + h[cI] * h_p1[cI - nx]
						   + m[cI] * h_p1[cI + nx * ny] + n[cI] * h_p1[cI - nx * ny]
						   - r[cI]
						   )
						   +
						   (1 - omegarelax) * h_p1[cI];
			}
		}
	}
	else 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			if(((k + j + i) % 2) != pass) {}
			else 
			{
				int cI = k * nx * ny + j * nx + i;
				h_p1[cI] = 	omegarelax 
							* 
							(
								(
									  e[cI] * h_p1[cI + 1]       +     f[cI] * h_p1[cI - 1]
								+ a * g[cI] * h_p1[cI + nx]      + a * h[cI] * h_p1[cI - nx]
								+ b * m[cI] * h_p1[cI + nx * ny] + b * n[cI] * h_p1[cI - nx * ny]
								- dx * dx * r[cI]
								)
								/
								(
								2.0 * (o[cI] + a * p[cI] + b * q[cI])
								)
							)
						   +
						   (1 - omegarelax) * h_p1[cI];
			}
		}
	}
}

void basic_iter
(
	double* e, double* f, double* g, double* h, double* m, double* n, double* r, 
	double* o, double* p, double* q,
	double a, double b, double omegarelax, double residual_reduction, double dx,
	double* h_p1, int nx, int ny, int nz
)
{
	if(USE_APRIORI_DENOMS) 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			int cI = k * nx * ny + j * nx + i;
			h_p1[cI] = (
						 e[cI] * h_p1[cI + 1]       + f[cI] * h_p1[cI - 1]
					   + g[cI] * h_p1[cI + nx]      + h[cI] * h_p1[cI - nx]
					   + m[cI] * h_p1[cI + nx * ny] + n[cI] * h_p1[cI - nx * ny]
					   - r[cI]
					   )
					   +
					   (1 - omegarelax) * h_p1[cI];
		}
	}
	else 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			int cI = k * nx * ny + j * nx + i;
			h_p1[cI] = 	omegarelax 
						* 
						(
							(
								  e[cI] * h_p1[cI + 1]       +     f[cI] * h_p1[cI - 1]
							+ a * g[cI] * h_p1[cI + nx]      + a * h[cI] * h_p1[cI - nx]
							+ b * m[cI] * h_p1[cI + nx * ny] + b * n[cI] * h_p1[cI - nx * ny]
							- dx * dx * r[cI]
							)
							/
							(
							2.0 * (o[cI] + a * p[cI] + b * q[cI])
							)
						)
					    +
					    (1 - omegarelax) * h_p1[cI];
		}		
	}
}

void basic_iter
(
	float* e, float* f, float* g, float* h, float* m, float* n, float* r,
	float* o, float* p, float* q,
	float a, float b, float omegarelax, float residual_reduction, float dx,
	float* h_p1, int nx, int ny, int nz
)
{
	if(USE_APRIORI_DENOMS) 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			int cI = k * nx * ny + j * nx + i;
			h_p1[cI] = (
						 e[cI] * h_p1[cI + 1]       + f[cI] * h_p1[cI - 1]
					   + g[cI] * h_p1[cI + nx]      + h[cI] * h_p1[cI - nx]
					   + m[cI] * h_p1[cI + nx * ny] + n[cI] * h_p1[cI - nx * ny]
					   - r[cI]
					   )
					   +
					   (1 - omegarelax) * h_p1[cI];
		}
	}
	else 
	{
		for(int k = 1; k < nz - 1; k++) for(int j = 1; j < ny - 1; j++) for(int i = 1; i < nx - 1; i++) 
		{
			int cI = k * nx * ny + j * nx + i;
			h_p1[cI] = 	omegarelax 
						* 
						(
							(
								  e[cI] * h_p1[cI + 1]       +     f[cI] * h_p1[cI - 1]
							+ a * g[cI] * h_p1[cI + nx]      + a * h[cI] * h_p1[cI - nx]
							+ b * m[cI] * h_p1[cI + nx * ny] + b * n[cI] * h_p1[cI - nx * ny]
							- dx * dx * r[cI]
							)
							/
							(
							2.0 * (o[cI] + a * p[cI] + b * q[cI])
							)
						)
					    +
					    (1 - omegarelax) * h_p1[cI];
		}		
	}
}

void apriori_denoms
(
	double* e, double* f, double* g, double* h, double* m, double* n,
	double* r, double* o, double* p, double* q, double a, double b,
	double omegarelax, double dx, int nx, int ny, int nz
)
{
	int d_size = nx * ny * nz;
	double* denom = (double*) malloc(d_size * sizeof(double));

	for(int i = 0; i < d_size; i++) {denom[i] = 1.0 / (2.0 * (o[i] + a * p[i] + b * q[i]));}

	for(int i = 0; i < d_size; i++) 
	{
		e[i] = omegarelax     * e[i] * denom[i];
		f[i] = omegarelax     * f[i] * denom[i];
		g[i] = omegarelax * a * g[i] * denom[i];
		h[i] = omegarelax * a * h[i] * denom[i];
		m[i] = omegarelax * b * m[i] * denom[i];
		n[i] = omegarelax * b * n[i] * denom[i];

		r[i] = omegarelax * dx * dx * r[i] * denom[i];
	}
}

void apriori_denoms
(
	float* e, float* f, float* g, float* h, float* m, float* n, 
	float* r, float* o, float* p, float* q, float a, float b, 
	float omegarelax, float dx, int nx, int ny, int nz
) 
{
	int d_size = nx * ny * nz;
	float* denom = (float*) malloc(d_size * sizeof(float));

	for(int i = 0; i < d_size; i++) {denom[i] = 1.0 / (2.0 * (o[i] + a * p[i] + b * q[i]));}

	for(int i = 0; i < d_size; i++) 
	{
		e[i] = omegarelax     * e[i] * denom[i];
		f[i] = omegarelax     * f[i] * denom[i];
		g[i] = omegarelax * a * g[i] * denom[i];
		h[i] = omegarelax * a * h[i] * denom[i];
		m[i] = omegarelax * b * m[i] * denom[i];
		n[i] = omegarelax * b * n[i] * denom[i];

		r[i] = omegarelax * dx * dx * r[i] * denom[i];
	}
}

