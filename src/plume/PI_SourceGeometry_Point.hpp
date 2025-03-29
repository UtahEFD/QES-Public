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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file SourcePoint.hpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once


#include "PI_SourceComponent.hpp"
#include "SourceGeometryPoint.h"

class PI_SourceGeometry_Point : public PI_SourceComponent
{
private:

  // position of the source
  float m_posX = -1.0f;
  float m_posY = -1.0f;
  float m_posZ = -1.0f;

protected:
  
public:
  // Default constructor
  PI_SourceGeometry_Point() : PI_SourceComponent()
  {
  }
  
  /*SourceGeometry_Point(HRRRData *hrrrInputData, WINDSGeneralData *WGD, int sid) : SourceGeometry(SourceShape::point)
  {  

    if (WGD->UTMZone == hrrrInputData->hrrrSourceUTMzone[sid]){
      posX = hrrrInputData->hrrrSourceUTMx[sid] - WGD->UTMx;;
    }else{
      int end_zone = 729400;
      int start_zone = 270570;
      int zone_diff = hrrrInputData->hrrrSourceUTMzone[sid] - WGD->UTMZone;
      posX = (hrrrInputData->hrrrSourceUTMx[sid] - start_zone) + (zone_diff - 1) * (end_zone - start_zone) + (end_zone - WGD->UTMx);
    }
    
    posY = hrrrInputData->hrrrSourceUTMy[sid] - WGD->UTMy;
    int site_i = posX / WGD->dx;
    int site_j = posY / WGD->dy;


    posZ = 8.0 + WGD->terrain[site_i + site_j *(WGD->nx - 1)];
    int site_k = (posZ / WGD->dz) + 1;
    }*/

  // destructor
  ~PI_SourceGeometry_Point() = default;


  void parseValues() override
  {

    parsePrimitive<float>(true, m_posX, "posX");
    parsePrimitive<float>(true, m_posY, "posY");
    parsePrimitive<float>(true, m_posZ, "posZ");
  }

  SourceComponent *create(QESDataTransport &data) override;
};
