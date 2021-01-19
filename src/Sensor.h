/*
 * QES-Winds
 *
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
 *
 */


#pragma once

/*
 * This is a collection of variables containing information relevant to
 * sensors read from an xml.
 */

#include <algorithm>
#include "util/ParseInterface.h"
#include "TimeSeries.h"

class WINDSInputData;
class WINDSGeneralData;

class Sensor : public ParseInterface
{
private:

  template<typename T>
  void _cudaCheck(T e, const char* func, const char* call, const int line);

public:


    float site_xcoord, site_ycoord;

    int site_coord_flag = 1;
    int site_UTM_zone;
    float site_UTM_x, site_UTM_y;
    float site_lon, site_lat;

    std::vector<TimeSeries*> TS;


    virtual void parseValues()
    {
      parsePrimitive<int>(false, site_coord_flag, "site_coord_flag");
      parsePrimitive<float>(false, site_xcoord, "site_xcoord");
      parsePrimitive<float>(false, site_ycoord, "site_ycoord");
      parsePrimitive<float>(false, site_UTM_x, "site_UTM_x");
      parsePrimitive<float>(false, site_UTM_y, "site_UTM_y");
      parsePrimitive<int>(false, site_UTM_zone, "site_UTM_zone");
      parsePrimitive<float>(false, site_lon, "site_lon");
      parsePrimitive<float>(false, site_lat, "site_lat");

      parseMultiElements<TimeSeries>(false, TS, "timeSeries");

    }

    void parseTree(pt::ptree t)
    {
        setTree(t);
        setParents("root");
        parseValues();
    }


    /**
     * @brief Computes the wind velocity profile using Barn's scheme
     * at the site's sensor
     *
     * This function takes in information for each site's sensor (boundary layer flag, reciprocal coefficient, surface
     * roughness and measured wind velocity and direction), generates wind velocity profile for each sensor and finally
     * utilizes Barns scheme to interplote velocity to generate the initial velocity field for the domain.
     */
    void inputWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD, int index, int solverType);


    /**
    * @brief Converts UTM to lat/lon and vice versa of the sensor coordiantes
    *
    */
    void UTMConverter (float rlon, float rlat, float rx, float ry, int UTM_PROJECTION_ZONE, int iway);

    /**
    * @brief Calculates the convergence value based on lat/lon input
    *
    */
    void getConvergence(float lon, float lat, int site_UTM_zone, float convergence);

    void BarnesInterpolationCPU (const WINDSInputData *WID, WINDSGeneralData *WGD, std::vector<std::vector<float>> u_prof, std::vector<std::vector<float>> v_prof);

    void BarnesInterpolationGPU (const WINDSInputData *WID, WINDSGeneralData *WGD, std::vector<std::vector<float>> u_prof, std::vector<std::vector<float>> v_prof);

};