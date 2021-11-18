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

/** @file SourceFullDomain.cpp 
 * @brief This class represents a specific source type. 
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#include "SourceFullDomain.hpp"


void SourceFullDomain::checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend)
{

  // notice that setting the variables as I am doing right now is not the standard way of doing this function
  xDomainStart = domainXstart;
  yDomainStart = domainYstart;
  zDomainStart = domainZstart;
  xDomainEnd = domainXend;
  yDomainEnd = domainYend;
  zDomainEnd = domainZend;


  if (xDomainStart > xDomainEnd) {
    std::cerr << "ERROR (SourceFullDomain::checkPosInfo): input xDomainStart is greater than input xDomainEnd! xDomainStart = \"" << xDomainStart
              << "\" xDomainEnd = \"" << xDomainEnd << "\"" << std::endl;
    exit(1);
  }
  if (yDomainStart > yDomainEnd) {
    std::cerr << "ERROR (SourceFullDomain::checkPosInfo): input yDomainStart is greater than input yDomainEnd! yDomainStart = \"" << yDomainStart
              << "\" yDomainEnd = \"" << yDomainEnd << "\"" << std::endl;
    exit(1);
  }
  if (zDomainStart > zDomainEnd) {
    std::cerr << "ERROR (SourceFullDomain::checkPosInfo): input zDomainStart is greater than input zDomainEnd! zDomainStart = \"" << zDomainStart
              << "\" zDomainEnd = \"" << zDomainEnd << "\"" << std::endl;
    exit(1);
  }

  // unfortunately there is no easy way to check that the input domain sizes are correct, so the code could potentially fail later on
  //  cause there is no easy checking method to be implemented here
}


int SourceFullDomain::emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles)
{
  // this function WILL fail if checkPosInfo() is not called, because for once checkPosInfo() acts to set the required data for using this function

  // release particle per timestep only if currTime is between m_releaseStartTime and m_releaseEndTime
  if (currTime >= m_rType->m_releaseStartTime && currTime <= m_rType->m_releaseEndTime) {
    std::random_device rd;//Will be used to obtain a seed for the random number engine
    std::mt19937 prng(rd());//Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> uniformDistr(0.0, 1.0);

    for (int pidx = 0; pidx < m_rType->m_parPerTimestep; pidx++) {

      Particle *cPar = new Particle();

      // generate uniform dist in domain
      cPar->xPos_init = uniformDistr(prng) * (xDomainEnd - xDomainStart) + xDomainStart;
      cPar->yPos_init = uniformDistr(prng) * (yDomainEnd - yDomainStart) + yDomainStart;
      cPar->zPos_init = uniformDistr(prng) * (zDomainEnd - zDomainStart) + zDomainStart;

      cPar->tStrt = currTime;

      cPar->sourceIdx = sourceIdx;

      emittedParticles.push_front(cPar);
    }
  }

  return emittedParticles.size();//m_rType->m_parPerTimestep;
}
