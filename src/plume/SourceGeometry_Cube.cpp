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

/** @file SourceCube.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourceGeometry_Cube.hpp"
#include "winds/WINDSGeneralData.h"
// #include "Interp.h"

void SourceCube::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{
  if (m_minX > m_maxX) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input minX is greater than input maxX! minX = \"" << m_minX
              << "\" maxX = \"" << m_maxX << "\"" << std::endl;
    exit(1);
  }
  if (m_minY > m_maxY) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input minY is greater than input maxY! minY = \"" << m_minY
              << "\" maxY = \"" << m_maxY << "\"" << std::endl;
    exit(1);
  }
  if (m_minZ > m_maxZ) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input minZ is greater than input maxZ! minZ = \"" << m_minZ
              << "\" maxZ = \"" << m_maxZ << "\"" << std::endl;
    exit(1);
  }

  if (m_minX < domainXstart) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input minX is outside of domain! minX = \"" << m_minX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (m_minY < domainYstart) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input minY is outside of domain! minY = \"" << m_minY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (m_minZ < domainZstart) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input minZ is outside of domain! minZ = \"" << m_minZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }

  if (m_maxX > domainXend) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input maxX is outside of domain! maxX = \"" << m_maxX
              << "\" domainXstart = \"" << domainXstart << "\" domainXend = \"" << domainXend << "\"" << std::endl;
    exit(1);
  }
  if (m_maxY > domainYend) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input maxY is outside of domain! maxY = \"" << m_maxY
              << "\" domainYstart = \"" << domainYstart << "\" domainYend = \"" << domainYend << "\"" << std::endl;
    exit(1);
  }
  if (m_maxZ > domainZend) {
    std::cerr << "ERROR (SourceCube::checkPosInfo): input maxZ is outside of domain! maxZ = \"" << m_maxZ
              << "\" domainZstart = \"" << domainZstart << "\" domainZend = \"" << domainZend << "\"" << std::endl;
    exit(1);
  }
}


int SourceCube::emitParticles(const float &dt,
                              const float &currTime,
                              std::list<Particle *> &emittedParticles)
{
  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime) {
    std::random_device rd;// Will be used to obtain a seed for the random number engine
    std::mt19937 prng(rd());// Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> uniformDistribution(0.0, 1.0);

    for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

      // Particle *cPar = new Particle();
      Particle *cPar = m_particleTypeFactory->Create(m_protoParticle);
      m_protoParticle->setParticleParameters(cPar);

      // generate uniform dist in domain
      cPar->xPos_init = uniformDistribution(prng) * (m_maxX - m_minX) + m_minX;
      cPar->yPos_init = uniformDistribution(prng) * (m_maxY - m_minY) + m_minY;
      cPar->zPos_init = uniformDistribution(prng) * (m_maxZ - m_minZ) + m_minZ;

      cPar->m = sourceStrength / m_rType->m_numPar;
      cPar->m_kg = cPar->m * (1.0E-3);
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
