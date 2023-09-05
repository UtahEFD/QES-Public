/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file SourceLine.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourceLine.hpp"
#include "winds/WINDSGeneralData.h"
// #include "Interp.h"

void SourceLine::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (posX_0 < domainXstart || posX_0 > domainXend) {
    std::cerr << "ERROR (SourceLine::checkPosInfo): input posX_0 is outside of domain! posX_0 = \"" << posX_0
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY_0 < domainYstart || posY_0 > domainYend) {
    std::cerr << "ERROR (SourceLine::checkPosInfo): input posY_0 is outside of domain! posY_0 = \"" << posY_0
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ_0 < domainZstart || posZ_0 > domainZend) {
    std::cerr << "ERROR (SourceLine::checkPosInfo): input posZ_0 is outside of domain! posZ_0 = \"" << posZ_0
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }

  if (posX_1 < domainXstart || posX_1 > domainXend) {
    std::cerr << "ERROR (SourceLine::checkPosInfo): input posX_1 is outside of domain! posX_1 = \"" << posX_1
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (posY_1 < domainYstart || posY_1 > domainYend) {
    std::cerr << "ERROR (SourceLine::checkPosInfo): input posY_1 is outside of domain! posY_1 = \"" << posY_1
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (posZ_1 < domainZstart || posZ_1 > domainZend) {
    std::cerr << "ERROR (SourceLine::checkPosInfo): input posZ_1 is outside of domain! posZ_1 = \"" << posZ_1
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}


int SourceLine::emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles)
{
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime) {
    for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

      // Particle *cPar = new Particle();
      Particle *cPar = m_particleTypeFactory->Create(m_protoParticle);

      // generate random point on line between m_pt0 and m_pt1
      double diffX = posX_1 - posX_0;
      double diffY = posY_1 - posY_0;
      double diffZ = posZ_1 - posZ_0;

      float t = drand48();

      // Now cPar is a generic particle, only created once (in setParticleType()).
      // If physical quantities should change per particle, the setParticleType() call should be moved here.
      cPar->xPos_init = posX_0 + t * diffX;
      cPar->yPos_init = posY_0 + t * diffY;
      cPar->zPos_init = posZ_0 + t * diffZ;
      // int cellId2d = interp->getCellId2d(cPar->xPos_init, cPar->yPos_init);
      // cPar->zPos_init = posZ_0 + t * diffZ + WGD->terrain[cellId2d];


      cPar->d = m_protoParticle->d;
      cPar->d_m = (1.0E-6) * m_protoParticle->d;
      cPar->rho = m_protoParticle->rho;
      cPar->depFlag = m_protoParticle->depFlag;
      cPar->decayConst = m_protoParticle->decayConst;
      cPar->c1 = m_protoParticle->c1;
      cPar->c2 = m_protoParticle->c2;

      cPar->m = sourceStrength / m_rType->m_numPar;
      cPar->m_kg = sourceStrength / m_rType->m_numPar * (1.0E-3);
      cPar->m_o = cPar->m;
      cPar->m_kg_o = cPar->m * (1.0E-3);
      // std::cout << " par type is: " << typeid(cPar).name() << " d = " << cPar->d << " m = " << cPar->m << " depFlag = " << cPar->depFlag << " vs = " << cPar->vs << std::endl;


      cPar->tStrt = currTime;

      cPar->sourceIdx = sourceIdx;

      emittedParticles.push_front(cPar);
    }
  }

  return emittedParticles.size();// m_rType->m_parPerTimestep;
}
