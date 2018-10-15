#pragma once

#include "URBInputData.h"
#include "SimulationParameters.h"
#include "Buildings.h"
#include "NonPolyBuilding.h"
#include "RectangularBuilding.h"
#include "Vector3.h"
#include "NetCDFData.h"
#include "Mesh.h"
#include "DTEHeightField.h"
#include "Cell.h"
#include "Sensor.h"
#include <math.h>
#include <vector>

#define MIN_S(x,y) ((x) < (y) ? (x) : (y))
#define MAX_S(x,y) ((x) > (y) ? (x) : (y))

/**< \class Solver
* This class declares and defines variables required for both solvers 
*/

class Solver
{
protected:
	int nx, ny, nz;		/**< number of cells */
	float dx, dy, dz;		/**< Grid resolution*/
    float dxy;		/**< Minimum value between dx and dy */
    std::vector<float> dzArray, zm;
    std::vector<float> x,y,z;
	int itermax;		/**< Maximum number of iterations */

	int num_sites;		/**< number of data entry sites */
	std::vector<int> site_blayer_flag;		/**< site boundary layer flag */
	std::vector<float> site_one_overL;		/**< Reciprocal Monin-Obukhov length (1/m) */
	std::vector<float> site_xcoord;		/**< location of the measuring site in x-direction */
	std::vector<float> site_ycoord;		/**< location of the measuring site in y-direction */
	std::vector<float> site_wind_dir;		/**< site wind wind direction */

	std::vector<float> site_z0;		/**< site surface roughness */
	std::vector<float> site_z_ref;		/**< measuring sensor height */
	std::vector<float> site_U_ref;		/**< site measured velocity */

    /// Declare coefficients for SOR solver
	std::vector<float> e;
	std::vector<float> f;
	std::vector<float> g;
	std::vector<float> h;
	std::vector<float> m;
	std::vector<float> n;

    /// Declaration of initial wind components (u0,v0,w0)
    std::vector<double> u0;
    std::vector<double> v0;
    std::vector<double> w0;
    
    std::vector<double> R;           /**< Divergence of initial velocity field */
  
    /// Declaration of final velocity field components (u,v,w)
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> w;

    /// Declaration of Lagrange multipliers
    std::vector<double> lambda;
	std::vector<double> lambda_old;
    std::vector<int> icellflag;        /// Cell index flag (0 = building, 1 = fluid)

    float max_velmag;
    std::vector<Building*> buildings;

    Mesh* mesh;
    int rooftopFlag;		/**< Rooftop flag */
    int upwindCavityFlag;		/**< Upwind cavity flag */
    int streetCanyonFlag;		/**< Street canyon flag */
    int streetIntersectionFlag;		/**< Street intersection flag */
    int wakeFlag;		/**< Wake flag */
    int sidewallFlag;		/**< Sidewall flag */

    Cell* cells;
    DTEHeightField* DTEHF;
	float z0;

    const int alpha1 = 1;        /**< Gaussian precision moduli */
    const int alpha2 = 1;        /**< Gaussian precision moduli */
    const float eta = pow(alpha1/alpha2, 2.0);
    const float A = pow(dx/dy, 2.0);
    const float B = eta*pow(dx/dz, 2.0);
    const float tol = 1e-9;     /**< Error tolerance */
    const float omega = 1.78;   /**< Over-relaxation factor */
    const float pi = 4.0f * atan(1.0);
    
	long numcell_cent;		/**< Total number of cell-centered values in domain */
	long numface_cent;		/**< Total number of face-centered values in domain */
	int icell_face;		/**< cell-face index */
	int icell_cent;		/**< cell-center index */  

	float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n;
	double *d_R;              //!> Divergence of initial velocity field
    double *d_lambda, *d_lambda_old;

    void printProgress (float percentage);

public:
	Solver(URBInputData* UID, DTEHeightField* DTEHF);

	virtual void solve(NetCDFData* netcdfDat, bool solveWind) = 0;

    void defineWalls(int* iCellFlag, float* n, float* m, float* f, float* e, float* h, float* g);
    void upWind(Building* build, int* iCellFlag, double* u0, double* v0, double* w0, float* z, float* zm);
    void reliefWake(NonPolyBuilding* build, float* u0, float* v0);
};
