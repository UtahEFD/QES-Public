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
#include "Canopies.h"
#include "Canopy.h"
#include "Cut_cell.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <limits>

using namespace std;

#define _USE_MATH_DEFINES
#define MIN_S(x,y) ((x) < (y) ? (x) : (y))
#define MAX_S(x,y) ((x) > (y) ? (x) : (y))

/**< \class Solver
* This class declares and defines variables required for both solvers
*
* There are several special member variables that should be accessible
* to all solvers.  They are declared in this class.
*/
class Solver
{
protected:

    //
    // CONSTANT VALUES
    //
    const int alpha1 = 1;        /**< Gaussian precision moduli */
    const int alpha2 = 1;        /**< Gaussian precision moduli */
    const float eta = pow(alpha1/alpha2, 2.0);
    const float A = pow(dx/dy, 2.0);
    const float B = eta*pow(dx/dz, 2.0);
    const float tol = 1.0e-9;     /**< Error tolerance -->>> very
                                   * small tol for float! */
    const float omega = 1.78f;   /**< Over-relaxation factor */
    const float pi = 4.0f * atan(1.0);

    const float vk = 0.4;			/// Von Karman's
                                                /// constant


    // SOLVER-based parameters
    int itermax;		/**< Maximum number of iterations */

    // General QUIC Domain Data needed in solvers
    int nx, ny, nz;		/**< number of cells */
    float dx, dy, dz;		/**< Grid resolution*/
    float dxy;		/**< Minimum value between dx and dy */
    std::vector<float> dzArray, zm;
    std::vector<float> x,y,z;

    Sensor* sensor;
    int num_sites;			/**< number of data entry sites */
    std::vector<int> site_blayer_flag;		/**< site boundary layer flag */
    std::vector<float> site_one_overL;		/**< Reciprocal Monin-Obukhov length (1/m) */
    std::vector<float> site_xcoord;			/**< location of the measuring site in x-direction */
    std::vector<float> site_ycoord;			/**< location of the measuring site in y-direction */
    std::vector<float> site_wind_dir;		/**< site wind wind direction */

    std::vector<float> site_z0;			/**< site surface roughness */
    std::vector<float> site_z_ref;		/**< measuring sensor height */
    std::vector<float> site_U_ref;		/**< site measured velocity */

    std::vector<float> site_canopy_H;
    std::vector<float> site_atten_coeff;

    float site_UTM_x, site_UTM_y, site_lat, site_lon;
    int site_coord_flag, site_UTM_zone;

    float UTMx, UTMy;       /**< Origin of QUIC domain UTM coordinates */
    int UTMZone;            /**< Domain UTM zone */
    float domain_rotation, theta;
    float convergence;

    int num_canopies;				/**< number of canopies */
    std::vector<float> atten;		/**< Attenuation coefficient */
    int landuse_flag;
    int landuse_veg_flag;
    int landuse_urb_flag;
    int lu_canopy_flag;
    std::vector<Building*> canopies;
    Canopy* canopy;


    /// Declaration of coefficients for SOR solver
    std::vector<float> e,f,g,h,m,n;

    /// Declaration of initial wind components (u0,v0,w0)
    std::vector<double> u0,v0,w0;

    std::vector<double> R;           /**< Divergence of initial velocity field */

    /// Declaration of final velocity field components (u,v,w)
    std::vector<double> u,v,w;
    std::vector<double> u_out,v_out,w_out;

    /// Declaration of Lagrange multipliers
    std::vector<double> lambda, lambda_old;
    std::vector<int> icellflag;        /// Cell index flag (0 = building/terrain, 1 = fluid)

    float z_ref;             /// Height of the measuring sensor (m)
    float U_ref;             /// Measured velocity at the sensor height (m/s)
    float z0;

    float max_velmag;
    std::vector<Building*> buildings;

    Mesh* mesh;
    int rooftopFlag;		/**< Rooftop flag */
    int upwindCavityFlag;		/**< Upwind cavity flag */
    int streetCanyonFlag;		/**< Street canyon flag */
    int streetIntersectionFlag;		/**< Street intersection flag */
    int wakeFlag;		/**< Wake flag */
    int sidewallFlag;		/**< Sidewall flag */
    int mesh_type_flag;		/**< mesh type (0 = Original QUIC/Stair-step, 1 = Cut-cell method) */

    Cell* cells;
    DTEHeightField* DTEHF;
    Cut_cell* cut_cell;

    long numcell_cent;		/**< Total number of cell-centered values in domain */
    long numface_cent;		/**< Total number of face-centered values in domain */
    int icell_face;		/**< cell-face index */
    int icell_cent;		/**< cell-center index */

    std::vector<std::vector<std::vector<float>>> x_cut;
    std::vector<std::vector<std::vector<float>>> y_cut;
    std::vector<std::vector<std::vector<float>>> z_cut;

    std::vector<std::vector<int>> num_points;
    std::vector<std::vector<float>> coeff;


    void printProgress (float percentage);

public:
    Solver(URBInputData* UID, DTEHeightField* DTEHF);

    virtual void solve(bool solveWind) = 0;

    virtual void outputDataFile() {}
    virtual void outputNetCDF( NetCDFData* netcdfDat ) {}

    /**
     * @brief
     *
     * This function takes in values necessary for cut-cell method for
     * buildings and then calculates the area fraction coefficients,
     * sets them to approperiate solver coefficients and finally sets
     * related coefficients to zero to define solid walls for non
     * cut-cells.
     */
    void defineWalls(float dx, float dy, float dz, int nx, int ny, int nz, int* icellflag, float* n, float* m,
                     float* f, float* e, float* h, float* g, std::vector<std::vector<std::vector<float>>> x_cut,
                     std::vector<std::vector<std::vector<float>>>y_cut,std::vector<std::vector<std::vector<float>>> z_cut,
                     std::vector<std::vector<int>> num_points, std::vector<std::vector<float>> coeff);
    /**
     * @brief
     *
     * This function takes in the icellflags set by setCellsFlag
     * function for stair-step method and sets related coefficients to
     * zero to define solid walls.
     */
    void defineWalls(float dx, float dy, float dz, int nx, int ny, int nz, int* icellflag, float* n, float* m,
                     float* f, float* e, float* h, float* g);
};
