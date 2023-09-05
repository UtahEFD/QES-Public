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

/** @file SourceCircle.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourceCircle.hpp"
#include "winds/WINDSGeneralData.h"
// #include "Interp.h"

void SourceCircle::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (radius < 0) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input radius is negative! radius = \"" << radius << "\"" << std::endl;
    exit(1);
  }

  if ((posX - radius) < domainXstart || (posX + radius) > domainXend) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input posX+radius is outside of domain! posX = \"" << posX << "\" radius = \"" << radius
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if ((posY - radius) < domainYstart || (posY + radius) > domainYend) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input posY+radius is outside of domain! posY = \"" << posY << "\" radius = \"" << radius
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if ((posZ - radius) < domainZstart || (posZ + radius) > domainZend) {
    std::cerr << "ERROR (SourceCircle::checkPosInfo): input posZ is outside of domain! posZ = \"" << posZ << "\" radius = \"" << radius
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}


int SourceCircle::emitParticles(const float dt,
                                const float currTime,
                                std::list<Particle *> &emittedParticles)
{
  // warning!!! this is still a point source! Need to work out the geometry details still
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime) {
    std::random_device rd;// Will be used to obtain a seed for the random number engine
    std::mt19937 prng(rd());// Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> uniformDistr(0.0, 1.0);
    for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

      // Particle *cPar = new Particle();


      Particle *cPar = m_particleTypeFactory->Create(m_protoParticle);
      m_protoParticle->setParticleParameters(cPar);
      
      cPar->xPos_init = posX;
      cPar->yPos_init = posY;
      cPar->zPos_init = posZ;

      cPar->m = sourceStrength / m_rType->m_numPar;
      cPar->m_kg = sourceStrength / m_rType->m_numPar * (1.0E-3);
      cPar->m_o = cPar->m;
      cPar->m_kg_o = cPar->m * (1.0E-3);

      cPar->tStrt = currTime;

      cPar->sourceIdx = sourceIdx;

      emittedParticles.push_front(cPar);
    }
  }

  return emittedParticles.size();// m_rType->m_parPerTimestep;
}
