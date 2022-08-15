/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file SimulationParameters.h */

#pragma once

#include <string>
#include "util/ParseInterface.h"
#include "util/ParseVector.h"
#include "util/Vector3.h"
#include "util/Vector3Int.h"
#include "DTEHeightField.h"
#include "WRFInput.h"
#include "Mesh.h"

/**
 * @class SimulationParameters
 * @brief Contains variables that define information
 * necessary for running the simulation.
 *
 * @sa ParseInterface
 */
class SimulationParameters : public ParseInterface
{
private:
public:
  Vector3Int domain; /**< :Number of cells in each direction: */
  Vector3 grid; /**< :Cell sizes in each direction: */
  int verticalStretching = 0; /**< :Defines grid type in z-direction (0-unifrom (default), 1-custom): */
  std::vector<float> dz_value; /**< :Array holding values of dz if the verticalStretching is 1: */
  int totalTimeIncrements; /**< :Total number of time steps: */
  int rooftopFlag = 1; /**< :Rooftop flag (0-none, 1-log profile (default), 2-vortex): */
  int upwindCavityFlag = 2; /**< :Upwind cavity flag (0-none, 1-Rockle, 2-MVP (default), 3-HMVP): */
  int streetCanyonFlag = 1; /**< :Street canyon flag (0-none, 1-Roeckle w/ Fackrel (default)): */
  int streetIntersectionFlag = 0; /**< :Street intersection flag (0-off, 1-on): */
  int wakeFlag = 2; /**< :Wake flag (0-none, 1-Rockle, 2-Modified Rockle (default), 3-Area Scaled): */
  int sidewallFlag = 1; /**< :Sidewall flag (0-off, 1-on (default)): */
  int logLawFlag = 0; /**< :Log Law flag to apply the log law (0-off (default), 1-on): */
  int maxIterations = 500; /**< :Maximum number of iterations (default = 500): */
  double tolerance = 1e-9; /**< :Convergence criteria, error threshold (default = 1e-9): */
  int meshTypeFlag = 0; /**< :Type of meshing scheme (0-Stair step (original QES) (default), 1-Cut-cell method: */
  float domainRotation = 0; /**< :Rotation angle of domain relative to true north: */
  int originFlag = 0; /**< :Origin flag (0- DEM coordinates (default), 1- UTM coordinates): */
  float UTMx; /**< :x component (m) of origin in UTM DEM coordinates (if originFlag = 1): */
  float UTMy; /**< :y component (m) of origin in UTM DEM coordinates (if originFlag = 1): */
  int UTMZone; /**< :UTM zone that domain located: */
  std::string UTMZoneLetter;
  float DEMDistancex = 0.0; /**< :x component (m) of origin in DEM coordinates (if originFlag = 0): */
  float DEMDistancey = 0.0; /**< :y component (m) of origin in DEM coordinates (if originFlag = 0): */
  float halo_x = 0.0; /**< :Halo region added to x-direction of domain (at the beginning and the end of domain) (meters): */
  float halo_y = 0.0; /**< :Halo region added to y-direction of domain (at the beginning and the end of domain) (meters): */
  float heightFactor = 1.0; /**< :Height factor multiplied by the building height read in from the shapefile (default = 1.0): */

  int readCoefficientsFlag = 0; /**< :Reading solver coefficients flag (0-calculate coefficients (default), 1-read coefficients from the file): */
  std::string coeffFile; /**< :Address to coefficients file location: */

  // DTE - digital elevation model details
  std::string demFile; /**< DEM file name */
  DTEHeightField *DTE_heightField = nullptr; /**< :document this: */
  Mesh *DTE_mesh; /**< :document this: */

  // //////////////////////////////////////////
  // WRF File Parameters
  // //////////////////////////////////////////
  //
  // Two use-cases are now supported:
  //
  // (1) Only a WRF file is supplied.
  // If only a WRF data output file is supplied in the XML, the
  // elevation, terrain model is acquired from the WRF Fire Mesh.
  // Metparmas related to stations/sensors are pulled from the wind
  // profile supplied by WRF.
  // * Issues:
  // - We are not yet checking if no fire mesh is specified.  If no
  // fire mesh is available, the terrain height could come from the
  // atmos mesh.
  //
  // (2) Both a DEM and a WRF file are supplied.  With both a DEM
  // and WRF file, the DEM will be used for creating the terrain and
  // the WRF file will only be used to extract stations/sensors
  // pulled from the wind profile in the WRF atmospheric mesh. Thus,
  // no terrain will be queried from the WRF file.
  //

  std::string wrfFile; /**< :document this: */
  WRFInput *wrfInputData = nullptr; /**< :document this: */
  int wrfSensorSample; /**< :document this: */
  bool wrfCoupling = false;

  enum DomainInputType {
    WRFOnly,
    WRFDEM,
    DEMOnly,
    UNKNOWN
  };
  DomainInputType m_domIType;

  SimulationParameters()
  {
    UTMx = 0.0;
    UTMy = 0.0;
    UTMZone = 0;
    UTMZoneLetter = "";
  }

  ~SimulationParameters()
  {
    // close the scanner
    if (DTE_heightField)
      DTE_heightField->closeScanner();
  }


  /**
   * :document this:
   */
  virtual void parseValues()
  {
    ParseVector<int> *domain_in;
    parseElement<ParseVector<int>>(false, domain_in, "domain");
    if (domain_in) {
      if (domain_in->size() == 3) {
        domain = { (*domain_in)[0], (*domain_in)[1], (*domain_in)[2] };
      } else {
        std::cerr << "[ERROR] unvalid number of argument for domain parameter" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    ParseVector<float> *grid_in;
    parseElement<ParseVector<float>>(false, grid_in, "cellSize");
    if (grid_in) {
      if (grid_in->size() == 3) {
        grid = { (*grid_in)[0], (*grid_in)[1], (*grid_in)[2] };
      } else {
        std::cerr << "[ERROR] unvalid number of argument for cellSize parameter" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    parsePrimitive<int>(false, verticalStretching, "verticalStretching");
    parseMultiPrimitives<float>(false, dz_value, "dz_value");
    parsePrimitive<int>(false, totalTimeIncrements, "totalTimeIncrements");
    parsePrimitive<int>(false, rooftopFlag, "rooftopFlag");
    parsePrimitive<int>(false, upwindCavityFlag, "upwindCavityFlag");
    parsePrimitive<int>(false, streetCanyonFlag, "streetCanyonFlag");
    parsePrimitive<int>(false, streetIntersectionFlag, "streetIntersectionFlag");
    parsePrimitive<int>(false, wakeFlag, "wakeFlag");
    parsePrimitive<int>(false, sidewallFlag, "sidewallFlag");
    parsePrimitive<int>(false, logLawFlag, "logLawFlag");
    parsePrimitive<int>(false, maxIterations, "maxIterations");
    parsePrimitive<double>(false, tolerance, "tolerance");
    parsePrimitive<int>(false, meshTypeFlag, "meshTypeFlag");
    parsePrimitive<float>(false, domainRotation, "domainRotation");
    parsePrimitive<int>(false, originFlag, "originFlag");
    parsePrimitive<float>(false, UTMx, "UTMx");
    parsePrimitive<float>(false, UTMy, "UTMy");
    parsePrimitive<int>(false, UTMZone, "UTMZone");
    parsePrimitive<std::string>(false, UTMZoneLetter, "UTMZoneLetter");
    parsePrimitive<float>(false, DEMDistancex, "DEMDistancex");
    parsePrimitive<float>(false, DEMDistancey, "DEMDistancey");
    parsePrimitive<float>(false, halo_x, "halo_x");
    parsePrimitive<float>(false, halo_y, "halo_y");
    parsePrimitive<float>(false, heightFactor, "heightFactor");
    parsePrimitive<int>(false, readCoefficientsFlag, "readCoefficientsFlag");

    coeffFile = "";
    parsePrimitive<std::string>(false, coeffFile, "COEFF");

    demFile = "";
    parsePrimitive<std::string>(false, demFile, "DEM");

    wrfFile = "";
    wrfSensorSample = 1;
    parsePrimitive<std::string>(false, wrfFile, "WRF");
    parsePrimitive<int>(false, wrfSensorSample, "WRFSensorSample");

    //
    // WRF coupling
    //
    parsePrimitive<bool>(false, wrfCoupling, "WRFCoupling");

    // Determine which use case to use for WRF/DEM combinations
    if ((demFile != "") && (wrfFile != "")) {
      // Both specified
      // DEM - read in terrain
      // WRF - retrieve wind profiles only
      m_domIType = WRFDEM;
    } else if ((demFile == "") && (wrfFile != "")) {
      // Only WRF Specified
      // WRF - pull terrain and retrieve wind profiles
      m_domIType = WRFOnly;
    } else if (demFile != "") {
      // Only DEM Specified - nothing set for WRF Input
      m_domIType = DEMOnly;
    } else {
      m_domIType = UNKNOWN;
    }

    if (demFile != "")
      demFile = QESfs::get_absolute_path(demFile);
    if (wrfFile != "")
      wrfFile = QESfs::get_absolute_path(wrfFile);
    if (coeffFile != "")
      coeffFile = QESfs::get_absolute_path(coeffFile);


    //
    // Process the data files based on the state determined above
    //
    if (m_domIType == WRFOnly) {
      //
      // WRF File is specified
      // Read in height field
      //
      // This step parses the WRF data, attempting to extract
      // the Fire mesh if it exists (the atm mesh, otherwise)
      // into a height field.
      //
      // We extract the dimensions of the fire (or atm) mesh for
      // nx and ny domain cells.
      //
      std::cout << "Processing WRF data for terrain and met param sensors from " << wrfFile << std::endl;
      wrfInputData = new WRFInput(wrfFile, UTMx, UTMy, UTMZone, UTMZoneLetter, 0, 0, wrfCoupling, wrfSensorSample);
      std::cout << "WRF input data processing completed." << std::endl;

      // Apply halo to wind profile locations -- halo units are
      // meters so can add to other coordinates/positions, such
      // as station coordinates.
      wrfInputData->applyHalotoStationData(halo_x, halo_y);

      // Debug to show where the stations are...
      // wrfInputData->dumpStationData();

      // In the current setup, grid may NOT be set... be careful
      // may need to initialize it here if nullptr is true for grid

      // utilize the wrf information to construct a
      // DTE_heightfield
      std::cout << "Constructing the DTE from WRF input heighfield..." << std::endl;

      DTE_heightField = new DTEHeightField(wrfInputData->fmHeight,
                                           std::tuple<int, int, int>(wrfInputData->fm_nx, wrfInputData->fm_ny, wrfInputData->fm_nz),
                                           std::tuple<float, float, float>(wrfInputData->fm_dx, wrfInputData->fm_dy, wrfInputData->fm_dz),
                                           halo_x,
                                           halo_y);

      // Make sure to pull the dx, dy and dz back from the WRF calculations
      grid[0] = wrfInputData->fm_dx;
      grid[1] = wrfInputData->fm_dy;
      grid[2] = wrfInputData->fm_dz;

      std::cout << "Dim: " << wrfInputData->fm_nx << " X " << wrfInputData->fm_ny << " X " << wrfInputData->fm_nz
                << " at " << grid[0] << " X " << grid[1] << " X " << grid[2] << std::endl;

      domain = { wrfInputData->fm_nx, wrfInputData->fm_ny, wrfInputData->fm_nz };
      DTE_heightField->setDomain(domain, grid);

      // need this to make sure domain sizes include halo
      int halo_x_WRFAddition = (int)floor(halo_x / wrfInputData->fm_dx);
      int halo_y_WRFAddition = (int)floor(halo_y / wrfInputData->fm_dy);

      std::cout << "Halo Addition to dimensions: " << halo_x_WRFAddition << " cells, " << halo_y_WRFAddition << " cells" << std::endl;

      domain[0] += 2 * halo_x_WRFAddition;
      domain[1] += 2 * halo_y_WRFAddition;

      // let WRF class know about the dimensions that the halo adds...
      wrfInputData->setHaloAdditions(halo_x_WRFAddition, halo_y_WRFAddition);

      std::cout << "domain size with halo borders: " << domain[0] << " x " << domain[1] << std::endl;

      DTE_mesh = new Mesh(DTE_heightField->getTris());
      std::cout << "Meshing of DEM complete\n";
    } else if (m_domIType == WRFDEM) {

      std::cout << "Reading DEM and processing WRF data for met param sensors from " << wrfFile << std::endl;

      // First read DEM as usual
      std::cout << "Extracting Digital Elevation Data from " << demFile << std::endl;

      std::cout << "Domain: " << domain[0] << ", " << domain[1] << ", " << domain[2] << std::endl;
      std::cout << "Grid: " << grid[0] << ", " << grid[1] << ", " << grid[2] << std::endl;
      DTE_heightField = new DTEHeightField(demFile,
                                           std::tuple<int, int, int>(domain[0], domain[1], domain[2]),
                                           std::tuple<float, float, float>(grid[0], grid[1], grid[2]),
                                           UTMx,
                                           UTMy,
                                           originFlag,
                                           DEMDistancex,
                                           DEMDistancey);
      assert(DTE_heightField);

      std::cout << "Forming triangle mesh...\n";
      DTE_heightField->setDomain(domain, grid);
      DTE_mesh = new Mesh(DTE_heightField->getTris());
      std::cout << "Mesh complete\n";

      // To proceed and cull sensors appropriately, we will need
      // the lower-left bounds from the DEM if UTMx and UTMy and
      // UTMZone are not all 0
      //
      // ??? Parse primitive should return true or false if a
      // value was parsed, rather than this
      // ??? can refactor that later.
      bool useUTM_for_DEMLocation = false;
      float uEps = 0.001;
      if (((UTMx > -uEps) && (UTMx < uEps)) && ((UTMy > -uEps) && (UTMy < uEps)) && (UTMZone == 0)) {
        useUTM_for_DEMLocation = true;
      }
      if (useUTM_for_DEMLocation) {
        std::cout << "UTM (" << UTMx << ", " << UTMy << "), Zone: " << UTMZone << " will be used as lower-left location for DEM." << std::endl;
      }

      // Note from Pete:
      // Normally, will want to see if UTMx and UTMy are valid
      // in the DEM and if so, pass that UTMx and UTMy into the
      // WRFInput.  Passing 0s for these should not likely cause
      // issues since we're either adding or subtracting these
      // values.

      // Then, read WRF File extracting ONLY the sensor data
      bool onlySensorData = true;
      float dimX = domain[0] * grid[0];
      float dimY = domain[1] * grid[1];
      std::cout << "dimX = " << dimX << ", dimY = " << dimY << std::endl;
      wrfInputData = new WRFInput(wrfFile, UTMx, UTMy, UTMZone, UTMZoneLetter, dimX, dimY, wrfSensorSample, wrfCoupling, onlySensorData);
      std::cout << "WRF Wind Velocity Profile Data processing completed." << std::endl;
    }

    else if (m_domIType == DEMOnly) {
      std::cout << "Extracting Digital Elevation Data from " << demFile << std::endl;
      DTE_heightField = new DTEHeightField(demFile,
                                           std::tuple<int, int, int>(domain[0], domain[1], domain[2]),
                                           std::tuple<float, float, float>(grid[0], grid[1], grid[2]),
                                           UTMx,
                                           UTMy,
                                           originFlag,
                                           DEMDistancex,
                                           DEMDistancey);
      assert(DTE_heightField);

      std::cout << "Forming triangle mesh...\n";
      DTE_heightField->setDomain(domain, grid);
      DTE_mesh = new Mesh(DTE_heightField->getTris());
      std::cout << "Mesh complete\n";
    } else {
      // No DEM, so make sure these are null
      DTE_heightField = nullptr;
      DTE_mesh = nullptr;
      wrfInputData = nullptr;
    }
  }
};
