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


#include "SourceGeometry.hpp"
#include "winds/WINDSGeneralData.h"
// #include "Particles.hpp"

class SourceGeometry_Point : public SourceGeometry
{
private:
  // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
  // guidelines for how to set these variables within an inherited source are given in SourceType.

protected:
public:
  double posX = -1.0;
  double posY = -1.0;
  double posZ = -1.0;
  // Default constructor
  SourceGeometry_Point() : SourceGeometry(SourceShape::point)
  {
  }
  
  SourceGeometry_Point(HRRRData *hrrrInputData, WINDSGeneralData *WGD, int sid) : SourceGeometry(SourceShape::point)
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

    

    //std::cout << "sid:   " << sid << std::endl;
    //std::cout << "posX:   " << posX << std::endl;
    //std::cout << "posY:   " << posY << std::endl;

    posZ = 8.0 + WGD->terrain[site_i + site_j *(WGD->nx - 1)];
    int site_k = (posZ / WGD->dz) + 1;
    if (sid == 28){
      std::cout << "posX:   " << posX << std::endl;
      std::cout << "posY:   " << posY << std::endl;
      std::cout << "site_i:   " << site_i << std::endl;
      std::cout << "site_j:   " << site_j << std::endl;
      std::cout << "site_k:   " << site_k << std::endl;
      std::cout << "posZ:   " << posZ << std::endl;
      std::cout << "WGD->terrain[site_i + site_j * (WGD->nx -1)]:   " << WGD->terrain[site_i + site_j * (WGD->nx-1)] << std::endl;
      std::cout << "WGD->u[site_i + site_j * WGD->nx + site_k * WGD->nx * WGD->ny]:   " << WGD->u[site_i + site_j * WGD->nx + site_k * WGD->nx * WGD->ny] << std::endl;
      std::cout << "WGD->u[site_i+1 + site_j * WGD->nx + site_k * WGD->nx * WGD->ny]:   " << WGD->u[site_i+1 + site_j * WGD->nx + site_k * WGD->nx * WGD->ny] << std::endl;
      std::cout << "WGD->v[site_i + site_j * WGD->nx + site_k * WGD->nx * WGD->ny]:   " << WGD->v[site_i + site_j * WGD->nx + site_k * WGD->nx * WGD->ny] << std::endl;
      std::cout << "WGD->v[site_i + (site_j+1) * WGD->nx + site_k * WGD->nx * WGD->ny]:   " << WGD->v[site_i + (site_j+1) * WGD->nx + site_k * WGD->nx * WGD->ny] << std::endl;
      std::cout << "WGD->icellflag[site_i + (site_j) * (WGD->nx-1) + site_k * (WGD->nx-1) * (WGD->ny-1)]:   " << WGD->icellflag[site_i + (site_j) * (WGD->nx-1) + site_k * (WGD->nx-1) * (WGD->ny-1)] << std::endl;
    }
  }

  // destructor
  ~SourceGeometry_Point() = default;


  void parseValues() override
  {
    parsePrimitive<double>(false, posX, "posX");
    parsePrimitive<double>(false, posY, "posY");
    parsePrimitive<double>(false, posZ, "posZ");
  }


  void checkPosInfo(const double &domainXstart,
                    const double &domainXend,
                    const double &domainYstart,
                    const double &domainYend,
                    const double &domainZstart,
                    const double &domainZend) override;

  void setInitialPosition(Particle *ptr) override;
};
