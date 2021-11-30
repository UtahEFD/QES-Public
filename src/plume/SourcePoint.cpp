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

/** @file SourcePoint.cpp
 * @brief This class represents a specific source type. 
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourcePoint.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

void SourcePoint::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (posX < domainXstart || posX > domainXend) {
    std::cerr << "ERROR (SourcePoint::checkPosInfo): input posX is outside of domain! posX = \"" << posX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY < domainYstart || posY > domainYend) {
    std::cerr << "ERROR (SourcePoint::checkPosInfo): input posY is outside of domain! posY = \"" << posY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ < domainZstart || posZ > domainZend) {
    std::cerr << "ERROR (SourcePoint::checkPosInfo): input posZ is outside of domain! posZ = \"" << posZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}

//template <class typeid(parType).name()>
int SourcePoint::emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles)
{
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime) {

    for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {
      //parType *cPar = new parType();
      //Particle *cPar = new Particle();
     
      auto cPar = particleTypeFactory.create(protoParticle->tag);
      
      cPar->xPos_init = posX;
      cPar->yPos_init = posY;
      cPar->zPos_init = posZ;

      cPar->d = protoParticle->d; 
      cPar->d_m = (1.0E-6)*protoParticle->d;
      cPar->rho = protoParticle->rho; 
      cPar->depFlag = protoParticle->depFlag; 
      
      cPar->m = sourceStrength/m_rType->m_numPar;
      cPar->m_kg = sourceStrength/m_rType->m_numPar * (1.0E-3); 
      
      std::cout << " par type is: " << cPar->tag << " d = " << cPar->d << " m = " << cPar->m << " depFlag = " << cPar->depFlag << " vs = " << cPar->vs << std::endl;

      
      cPar->tStrt = currTime;

      cPar->sourceIdx = sourceIdx;

      emittedParticles.push_front(cPar);
    }
  }

  return emittedParticles.size();//m_rType->m_parPerTimestep;
}
