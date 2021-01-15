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


#include "util/ParseInterface.h"

class URBInputData;
class URBGeneralData;


class TimeSeries : public ParseInterface
{
private:

public:

  int site_blayer_flag = 1;
  float site_z0;
  std::vector<float> site_wind_dir, site_z_ref, site_U_ref;
  float site_one_overL;
  float site_canopy_H, site_atten_coeff;



  virtual void parseValues()
  {
    parsePrimitive<int>(false, site_blayer_flag, "boundaryLayerFlag");
    parsePrimitive<float>(true, site_z0, "siteZ0");
    parsePrimitive<float>(true, site_one_overL, "reciprocal");
    parseMultiPrimitives<float>(true, site_z_ref, "height");
    parseMultiPrimitives<float>(true, site_U_ref, "speed");
    parseMultiPrimitives<float>(true, site_wind_dir, "direction");
    parsePrimitive<float>(false, site_canopy_H, "canopyHeight");
    parsePrimitive<float>(false, site_atten_coeff, "attenuationCoefficient");
  }

};
