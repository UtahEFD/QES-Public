#pragma once

#include <vector>
#include <netcdf>
#include <cmath>


#define _USE_MATH_DEFINES
#define MIN_S(x,y) ((x) < (y) ? (x) : (y))
#define MAX_S(x,y) ((x) > (y) ? (x) : (y))

#include "WINDSInputData.h"
#include "Building.h"
#include "Canopy.h"
#include "LocalMixing.h"
#include "LocalMixingDefault.h"
#include "LocalMixingNetCDF.h"
#include "LocalMixingSerial.h"
#include "Mesh.h"
#include "DTEHeightField.h"
#include "Cut_cell.h"
#include "Wall.h"
#include "NetCDFInput.h"

#ifdef HAS_OPTIX
#include "OptixRayTrace.h"
#endif

using namespace netCDF;
using namespace netCDF::exceptions;

class WINDSInputData;

class WINDSGeneralData {
public:
    WINDSGeneralData();
    WINDSGeneralData(const WINDSInputData* WID);
    ~WINDSGeneralData();

    void mergeSort( std::vector<float> &effective_height,
                    std::vector<Building*> allBuildingsV,
                    std::vector<int> &building_id );
    
    void applyParametrizations(const WINDSInputData* UID);

    /*!
    * This function is being called from the plantInitial function
    * and uses the bisection method to find the displacement height
    * of the canopy.
    */
    float canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

    /**
    * @brief
    *
    * This function saves user-defined data to file
    */
    void save();

    ////////////////////////////////////////////////////////////////////////////
    //////// Variables and constants needed only in other functions-- Behnam
    //////// This can be moved to a new class (WINDSGeneralData)
    ////////////////////////////////////////////////////////////////////////////
    const float pi = 4.0f * atan(1.0);
    const float vk = 0.4;			/// Von Karman's
                                                /// constant
    float cavity_factor, wake_factor;
    float lengthf_coeff;
    float theta;

    // General QUIC Domain Data
    int nx, ny, nz;		/**< number of cells */
    float dx, dy, dz;		/**< Grid resolution*/
    float dxy;		/**< Minimum value between dx and dy */

    long numcell_cout;
    long numcell_cout_2d;
    long numcell_cent;       /**< Total number of cell-centered values in domain */
    long numcell_face;       /**< Total number of face-centered values in domain */
    std::vector<size_t> start;
    std::vector<size_t> count;

    std::vector<float> z0_domain_u, z0_domain_v;

    std::vector<int> ibuilding_flag;
    std::vector<int> building_id;
    std::vector<Building*> allBuildingsV;

    float z0;           // In wallLogBC

    std::vector<float> dz_array;
    std::vector<float> x,y,z;
    std::vector<float> z_face;
    //std::vector<float> x_out,y_out,z_out;

    /// Declaration of coefficients for SOR solver
    std::vector<float> e,f,g,h,m,n;

    // The following are mostly used for output
    std::vector<int> icellflag;  /**< Cell index flag (0 = Building, 1 = Fluid, 2 = Terrain, 3 = Upwind cavity
                                                       4 = Cavity, 5 = Farwake, 6 = Street canyon, 7 = Building cut-cells,
                                                       8 = Terrain cut-cells, 9 = Sidewall, 10 = Rooftop,
                                                       11 = Canopy vegetation, 12 = Fire) */

    std::vector<float> building_volume_frac, terrain_volume_frac;
    std::vector<float> terrain;
    std::vector<int> terrain_id;      // Sensor function
                                      // (inputWindProfile)

    std::vector <float> base_height;      // Base height of buildings
    std::vector <float> effective_height;            // Effective height of buildings

    // Initial wind conditions
    /// Declaration of initial wind components (u0,v0,w0)
    std::vector<float> u0,v0,w0;

    /// Declaration of final velocity field components (u,v,w)
    std::vector<float> u,v,w;

    // local Mixing class and data
    LocalMixing* localMixing;
    std::vector<double> mixingLengths;

    // Sensor* sensor;      may not need this now

    int id;

    // [FM Feb.28.2020] there 2 variables are not used anywhere
    //std::vector<float> site_canopy_H;
    //std::vector<float> site_atten_coeff;

    float convergence;
    // Canopy functions

    std::vector<float> canopy_atten;		/**< Canopy attenuation coefficient */
    std::vector<float> canopy_top;		  /**< Canopy height */
    std::vector<int> canopy_top_index;		  /**< Canopy top index */
    std::vector<float> canopy_z0;		  /**< Canopy surface roughness */
    std::vector<float> canopy_ustar;		  /**< Velocity gradient at the top of canopy */
    std::vector<float> canopy_d;		  /**< Canopy displacement length */

    //Canopy* canopy;

    float max_velmag;         // In polygonWake

    // In getWallIndices and wallLogBC
    std::vector<int> wall_right_indices;     /**< Indices of the cells with wall to right boundary condition */
    std::vector<int> wall_left_indices;      /**< Indices of the cells with wall to left boundary condition */
    std::vector<int> wall_above_indices;     /**< Indices of the cells with wall above boundary condition */
    std::vector<int> wall_below_indices;     /**< Indices of the cells with wall bellow boundary condition */
    std::vector<int> wall_back_indices;      /**< Indices of the cells with wall in back boundary condition */
    std::vector<int> wall_front_indices;     /**< Indices of the cells with wall in front boundary condition */

    Mesh* mesh;           // In terrain functions

    Cell* cells;
    // bool DTEHFExists = false;
    Cut_cell cut_cell;
    Wall *wall;

    NetCDFInput* NCDFInput;
    int ncnx, ncny, ncnz, ncnt;


    // Building cut-cell (rectangular building)
    std::vector<std::vector<std::vector<float>>> x_cut;
    std::vector<std::vector<std::vector<float>>> y_cut;
    std::vector<std::vector<std::vector<float>>> z_cut;
    std::vector<std::vector<int>> num_points;
    std::vector<std::vector<float>> coeff;

private:

};
