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

/** @file ReleaseType_continuous.hpp
 * @brief This class represents a specific release type.
 *
 * @note Child of ReleaseType
 * @sa ReleaseType
 */

#pragma once

#include "PI_ReleaseType.hpp"
#include "winds/WINDSGeneralData.h"

class PI_ReleaseType_instantaneous : public PI_ReleaseType
{
private:
  // note that this also inherits data members
  // ParticleReleaseType m_rType, int m_particlePerTimestep, float m_releaseStartTime,
  // float m_releaseEndTime, and int m_numPar from ReleaseType.
  // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

  float m_releaseTime = 0;
  int m_numPar = 0;
  float m_totalMass = 0;

protected:
public:
  // Default constructor
  PI_ReleaseType_instantaneous() : PI_ReleaseType()
  {
  }

  /*ReleaseType_instantaneous(HRRRData *hrrrInputData, int sid) : ReleaseType(ParticleReleaseType::instantaneous)
  {
    float hrrrDx = 3000;
    float hrrrDy = 3000;
    float hrrrDz = 2;
    float QESdx = 100;
    float QESdy = 100;
    float QESdz = 5;
    float particleMass = 1.8*pow(10, -12);
    
    if (hrrrInputData->hrrrC[sid] > 0){
      m_numPar = ((hrrrInputData->hrrrC[sid]/pow(10, 9)))/particleMass;
    }else{
      m_numPar = 0;
    }
    }*/

  // destructor
  ~PI_ReleaseType_instantaneous() override = default;

  void parseValues() override;
  void initialize(const float &timestep) override;
  SourceReleaseController *create(QESDataTransport &data) override;

};
