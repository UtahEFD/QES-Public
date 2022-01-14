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

/** @file Sensor.h */

#pragma once

#include <algorithm>
#include "util/ParseInterface.h"
#include "TimeSeries.h"

class WINDSInputData;
class WINDSGeneralData;

/**
 * @class Sensor
 * @brief Collection of variables containing information relevant to
 * sensors read from an xml.
 *
 * @sa ParseInterface
 * @sa TimeSeries
 */
class Sensor : public ParseInterface
{
private:
  /**
   * :document this:
   *
   * @param e :document this:
   * @param func :document this:
   * @param call :document this:
   * @param line :document this:
   */
  template<typename T>
  void _cudaCheck(T e, const char *func, const char *call, const int line);

public:
  Sensor()
  {
  }

  Sensor(const std::string fileName)
  {
    pt::ptree tree;

    try {
      pt::read_xml(fileName, tree);
    } catch (boost::property_tree::xml_parser::xml_parser_error &e) {
      std::cerr << "Error reading tree in" << fileName << "\n";
      exit(EXIT_FAILURE);
    }

    parseTree(tree);
  }


  ///@{
  /** Location of the sensor in QES domain */
  float site_xcoord, site_ycoord;
  ///@}

  int site_coord_flag = 1; /**< Sensor site coordinate system (1=QES (default), 2=UTM, 3=Lat/Lon) */
  int site_UTM_zone; /**< UTM zone of the sensor site (if site_coord_flag = 2) */

  ///@{
  /** x and y components of site coordinate in UTM (if site_coord_flag = 2) */
  float site_UTM_x, site_UTM_y;
  ///@}

  ///@{
  /** x and y components of site coordinate in Latitude and Longitude (if site_coord_flag = 3) */
  float site_lon, site_lat;
  ///@}

  std::vector<TimeSeries *> TS; /**< array of timestep informastion for a sensor */

  /**
   * This function parses the information in the xml file specified in the sensor section,
   * process them and assign them to the approperiate variables.
   */
  virtual void parseValues()
  {
    parsePrimitive<int>(true, site_coord_flag, "site_coord_flag");
    parsePrimitive<float>(false, site_xcoord, "site_xcoord");
    parsePrimitive<float>(false, site_ycoord, "site_ycoord");
    parsePrimitive<float>(false, site_UTM_x, "site_UTM_x");
    parsePrimitive<float>(false, site_UTM_y, "site_UTM_y");
    parsePrimitive<int>(false, site_UTM_zone, "site_UTM_zone");
    parsePrimitive<float>(false, site_lon, "site_lon");
    parsePrimitive<float>(false, site_lat, "site_lat");
    parseMultiElements<TimeSeries>(false, TS, "timeSeries");
  }

  /**
   * :document this:
   */
  void parseTree(pt::ptree t)
  {
    setTree(t);
    setParents("root");
    parseValues();
  }


  /**
   * Computes the wind velocity profile using Barnes scheme at the site's sensor.
   *
   *
   * Takes in information for each site's sensor (boundary layer flag, reciprocal coefficient, surface
   * roughness and measured wind velocity and direction), generates wind velocity profile for each sensor and finally
   * utilizes Barnes scheme to interplote velocity to generate the initial velocity field for the domain.
   *
   * @param WID a pointer instance to the class WINDSInputData that contains information read in from input files
   * @param WGD a pointer instance to the class WINDSGeneralData that contains data required to run QES-Winds
   * @param index timestep counter
   * @param solverType type of the solver specified by user
   */
  void inputWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD, int index, int solverType);


  /**
   * Converts UTM to lat/lon and vice versa of the sensor coordiantes.
   *
   * @param rlon :document this:
   * @param rlat :document this:
   * @param rx :document this:
   * @param ry :document this:
   * @param UTM_PROJECTION_ZONE :document this:
   * @param iway :document this:
   */
  void UTMConverter(float rlon, float rlat, float rx, float ry, int UTM_PROJECTION_ZONE, int iway);

  /**
   * Calculates the convergence value based on lat/lon input.
   *
   * @param lon :document this:
   * @param lat :document this:
   * @param site_UTM_zone :document this:
   * @param convergense :document this:
   */
  void getConvergence(float lon, float lat, int site_UTM_zone, float convergence);

  /**
   * This function utilizes Barnes scheme to interplote velocity to generate the initial velocity field for the domain.
   * This function is called when the CPU solver is specified.
   *
   * @param WID a pointer instance to the class WINDSInputData that contains information read in from input files
   * @param WGD a pointer instance to the class WINDSGeneralData that contains data required to run QES-Winds
   * @param u_prof u component of the velocity profile created by the sensor information
   * @param v_prof v component of the velocity profile created by the sensor information
   */
  void BarnesInterpolationCPU(const WINDSInputData *WID, WINDSGeneralData *WGD, std::vector<std::vector<float>> u_prof, std::vector<std::vector<float>> v_prof, int num_sites, std::vector<int> available_sensor_id);

  /**
   * This function utilizes Barnes scheme to interplote velocity to generate the initial velocity field for the domain.
   * This function is called when one of the GPU solvers is specified.
   *
   * @param WID a pointer instance to the class WINDSInputData that contains information read in from input files
   * @param WGD a pointer instance to the class WINDSGeneralData that contains data required to run QES-Winds
   * @param u_prof u component of the velocity profile created by the sensor information
   * @param v_prof v component of the velocity profile created by the sensor information
   * @param site_id flatten id of the sensor location in 2D domain region
   */
  void BarnesInterpolationGPU(const WINDSInputData *WID, WINDSGeneralData *WGD, std::vector<std::vector<float>> u_prof, std::vector<std::vector<float>> v_prof, std::vector<int> site_id, int num_sites, std::vector<int> available_sensor_id);
};
