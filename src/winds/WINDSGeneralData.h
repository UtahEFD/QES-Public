/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file WINDSGeneralData.h */

#pragma once

#include <vector>
#include <netcdf>
#include <cmath>


#define _USE_MATH_DEFINES
#define MIN_S(x, y) ((x) < (y) ? (x) : (y))
#define MAX_S(x, y) ((x) > (y) ? (x) : (y))

#include "WINDSInputData.h"

#include "qes/Domain.h"

#include "WindProfilerType.h"
#include "WindProfilerWRF.h"
#include "WindProfilerBarnCPU.h"
#include "WindProfilerBarnGPU.h"
#include "WindProfilerHRRR.h"

#include "Building.h"
#include "Canopy.h"
#include "CanopyElement.h"

#include "LocalMixing.h"
#include "LocalMixingDefault.h"
#include "LocalMixingNetCDF.h"
#include "LocalMixingSerial.h"

#include "DTEHeightField.h"
#include "CutCell.h"
#include "Wall.h"

// #include "util/Mesh.h"
#include "util/NetCDFInput.h"
#include "util/QEStime.h"
#include "HRRRData.h"

#ifdef HAS_OPTIX

// Needed to ensure that std::max and std::min are available
// since the optix.h headers eventually include windows.h on WIN32
// systems.  Windows overrides max and min and the following will
// force that off and thus allow std::max and std::min to work as
// desired.
#ifdef WIN32
#define NOMINMAX
#endif

#include "OptixRayTrace.h"
#endif

using namespace netCDF;
using namespace netCDF::exceptions;

struct WINDSDeviceData
{
  float *u;
  float *v;
  float *w;
};

class WINDSInputData;

/**
 * @class WindsGeneralData
 * @brief :document this:
 */
class WINDSGeneralData
{
private:
  WINDSGeneralData() : domain(1, 1, 1, 1, 1, 1) {}// do not allow empty domain to be created

public:
  explicit WINDSGeneralData(qes::Domain domain_in);
  WINDSGeneralData(const WINDSInputData *WID, qes::Domain domain_in, int solverType);
  explicit WINDSGeneralData(const std::string &inputFile);

  virtual ~WINDSGeneralData() = default;

  void mergeSort(std::vector<float> &effective_height,
                 std::vector<int> &building_id);

  void mergeSortTime(std::vector<QEStime> &sensortime,
                     std::vector<int> &sensortime_id);

  void applyWindProfile(const WINDSInputData *, int, int);

  void applyParametrizations(const WINDSInputData *);

  void downscaleHRRR(const WINDSInputData *);
  // void applyParametrizations(const WINDSInputData*);

  QEStime nextTimeInstance(const int &, const float &);

  void printTimeProgress(int);

  void resetICellFlag();
  bool isSolid(const int &);
  bool isCanopy(const int &);
  bool isTerrain(const int &);
  bool isBuilding(const int &);
  /**
   * Uses bisection method to find the displacement height of the canopy.
   *
   * @note Called within plantInitial method.
   *
   * @param ustar :document this:
   * @param z0 :document this:
   * @param canopy_top :document this:
   * @param canopy_atten :document this:
   * @param vk :document this:
   * @param psi_m :document this:
   */
  float canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

  /**
   * Saves user-defined data to file.
   */
  void save();
  void loadNetCDFData(int);

  ////////////////////////////////////////////////////////////////////////////
  //////// Variables and constants needed only in other functions-- Behnam
  //////// This can be moved to a new class (WINDSGeneralData)
  ////////////////////////////////////////////////////////////////////////////
  const float pi = 4.0f * atan(1.0); /**< pi constant */
  const float vk = 0.4; /**< Von Karman's constant */

  float cavity_factor = 1.0; /**< :document this: */
  float wake_factor = 0.0; /**< :document this: */
  float lengthf_coeff = 1.5; /**< :document this: */
  float theta{}; /**< :document this: */
  // int icell_face;
  // int icell_cent;

  // General QES Domain Data -- winds does not create this... provide const & in constructor of WINDS...
  qes::Domain domain;

  ///@{
  /** Number of cells */
  // int nx, ny, nz;
  ///@}
  ///@{
  /** Grid resolution */
  // float dx, dy, dz;
  ///@}
  // float dxy; /**< Minimum value between dx and dy */

  int wrf_nx{}, wrf_ny{};

  float halo_x{}, halo_y{};
  int halo_index_x{}, halo_index_y{};
  
  float UTMx, UTMy;
  int UTMZone;

  // long numcell_cout{}; /**< :document this: */
  // long numcell_cout_2d; /**< :document this: */
  // long numcell_cent; /**< Total number of cell-centered values in domain */
  // long numcell_face; /**< Total number of face-centered values in domain */

  // std::vector<size_t> start; /**< :document this: */
  // std::vector<size_t> count; /**< :document this: */

  ///@{
  /** :Values of z0 for u and v components: */
  std::vector<float> z0_domain_u, z0_domain_v;
  ///@}

  std::vector<int> ibuilding_flag; /**< :Building number flag: */
  std::vector<int> building_id; /**< :Building ID: */
  std::vector<Building *> allBuildingsV; /**< :Vector contains all of the building elements: */

  float z0{}; /**< In wallLogBC */

  // std::vector<float> dz_array; /**< :Array contain dz values: */
  ///@{
  /** :Location of center of cell in x,y and z directions: */
  // std::vector<float> x, y, z;
  ///@}
  // std::vector<float> z_face; /**< :Location of the bottom face of the cell in z-direction: */

  std::vector<QEStime> sensortime; /**< :document this: */
  std::vector<int> sensortime_id;
  // FM TEMPORARY!!!!!!!!!!!!!
  std::vector<int> time_id;

  // time variables
  int nt{}; /**< :document this: */
  int totalTimeIncrements{}; /**< :document this: */
  std::vector<float> dt_array; /**< :document this: */
  std::vector<QEStime> timestamp; /**< :document this: */


  ///@{
  /** Coefficient for SOR solver */
  std::vector<float> e, f, g, h, m, n;
  ///@}

  // The following are mostly used for output
  std::vector<int> icellflag;
  /**< Cell index flag (0 = Building, 1 = Fluid, 2 = Terrain, 3 = Upwind cavity
                        4 = Cavity, 5 = Farwake, 6 = Street canyon, 7 = Building cut-cells,
                        8 = Terrain cut-cells, 9 = Sidewall, 10 = Rooftop,
                        11 = Canopy vegetation, 12 = Fire) */
  std::vector<int> icellflag_initial;
  std::vector<int> icellflag_footprint;

  ///@{
  /** :document this: */
  std::vector<float> building_volume_frac, terrain_volume_frac;
  ///@}

  ///@{
  /** Normal components of the cut surface for the solid elements (Building or Terrain) */
  std::vector<float> ni, nj, nk;
  ///@}
  ///@{
  /** Tangential components of the cut surface for the solid elements (Building or Terrain) */
  std::vector<float> ti, tj, tk;
  ///@}
  std::vector<float> wall_distance; /**< :Distance of the cell center from the cut face: */
  std::vector<int> center_id; /**< :Defines whether a cell center is inside a solid (0) or air (1): */
  std::vector<float> terrain; /**< :document this: */
  std::vector<int> terrain_id;
  std::vector<int> terrain_face_id; /**< Sensor function (inputWindProfile) */

  std::vector<float> base_height; /**< Base height of buildings */
  std::vector<float> effective_height; /**< Effective height of buildings */

  // Initial wind conditions
  ///@{
  /** Declaration of initial wind components (u0,v0,w0) */
  std::vector<float> u0, v0, w0;
  // maintain a separate QESWindData for u0,v0,w0
  ///@}

  ///@{
  /** Declaration of final velocity field components (u,v,w) */
  std::vector<float> u, v, w;

  /// it really needs to use SimDataAccess.windData for the u, v, w....
  ///@}

  // local Mixing class and data
  LocalMixing *localMixing{}; /**< :document this: */
  std::vector<double> mixingLengths; /**< :document this: */

  // HRRR Input class
  HRRRData *hrrrInputData;
  std::vector<int> nearest_site_id;
  std::vector<std::vector<int>> closest_site_ids;
  // Sensor* sensor;      may not need this now

  // wind profiler class
  WindProfilerType *windProfiler{}; /**< pointer to the wind profiler class, used to interp wind */

  int id{}; /**< :document this: */

  // [FM Feb.28.2020] there 2 variables are not used anywhere
  // std::vector<float> site_canopy_H;
  // std::vector<float> site_atten_coeff;

  float convergence{}; /**< :document this: */

  // Canopy functions
  // std::vector<float> canopy_atten;   /**< Canopy attenuation coefficient */
  // std::vector<float> canopy_top;	   /**< Canopy height */
  // std::vector<int> canopy_top_index; /**< Canopy top index */
  // std::vector<float> canopy_z0;	   /**< Canopy surface roughness */
  // std::vector<float> canopy_ustar;   /**< Velocity gradient at the top of canopy */
  // std::vector<float> canopy_d;	   /**< Canopy displacement length */

  Canopy *canopy{}; /**< :document this: */


  float max_velmag{}; /**< In polygonWake */

  // In getWallIndices and wallLogBC
  std::vector<int> wall_right_indices; /**< Indices of the cells with wall to right boundary condition */
  std::vector<int> wall_left_indices; /**< Indices of the cells with wall to left boundary condition */
  std::vector<int> wall_above_indices; /**< Indices of the cells with wall above boundary condition */
  std::vector<int> wall_below_indices; /**< Indices of the cells with wall bellow boundary condition */
  std::vector<int> wall_back_indices; /**< Indices of the cells with wall in back boundary condition */
  std::vector<int> wall_front_indices; /**< Indices of the cells with wall in front boundary condition */
  std::vector<int> wall_indices; /**< Indices of the cells with wall on at least one side */

  Mesh *mesh{}; /**< In Terrain functions */

  Cell *cells{}; /**< :document this: */
  // bool DTEHFExists = false;
  CutCell *cut_cell; /**< :document this: */
  Wall *wall; /**< :document this: */

  // NetCDFInput* NCDFInput;     /**< :document this: */
  ///@{
  /** :document this: */
  // int ncnx, ncny, ncnz, ncnt;
  ///@}

  ///@{
  /** Building cut-cell (rectangular building) */
  std::vector<std::vector<std::vector<float>>> x_cut;
  std::vector<std::vector<std::vector<float>>> y_cut;
  std::vector<std::vector<std::vector<float>>> z_cut;
  std::vector<std::vector<int>> num_points;
  std::vector<std::vector<float>> coeff;
  ///@}

  WINDSDeviceData d_data;
  void allocateDevice();
  void freeDevice();
  void copyDataToDevice();
  void copyDataFromDevice();

private:
  // input: store here for multiple time instance.
  NetCDFInput *input{}; /**< :document this: */

protected:
  // void defineHorizontalGrid();
  // void defineVerticalGrid();
  // void defineVerticalStretching(const float &);
  // void defineVerticalStretching(const std::vector<float> &);

  void allocateMemory();
};

inline bool WINDSGeneralData::isCanopy(const int &cellID)
{
  return (icellflag[cellID] == 20 || icellflag[cellID] == 22 || icellflag[cellID] == 24 || icellflag[cellID] == 28);
}

inline bool WINDSGeneralData::isBuilding(const int &cellID)
{
  return (icellflag[cellID] == 0);
}

inline bool WINDSGeneralData::isTerrain(const int &cellID)
{
  return (icellflag[cellID] == 2);
}

inline bool WINDSGeneralData::isSolid(const int &cellID)
{
  return (icellflag[cellID] == 0 || icellflag[cellID] == 2);
}
