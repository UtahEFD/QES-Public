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

/** @file SourceFullDomain.hpp
 * @brief This class represents a specific source type. 
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once


#include "SourceType.hpp"
#include "winds/WINDSGeneralData.h"
//#include "Particles.hpp"

class SourceFullDomain : public SourceType
{
private:
  // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
  // guidelines for how to set these variables within an inherited source are given in SourceType.

  // this source is a bit weird because the domain size has to be obtained after the input parser.
  //  this would mean either doing a function call unique to this source to supply the required data during the dispersion constructor
  //  or by using checkPosInfo() differently than it is normally intended to set the domain size variables
  double xDomainStart;
  double yDomainStart;
  double zDomainStart;
  double xDomainEnd;
  double yDomainEnd;
  double zDomainEnd;
  double sourceStrength = 0.0;// total mass released (g)
protected:
public:
  // Default constructor
  SourceFullDomain()
  {
  }

  // destructor
  ~SourceFullDomain()
  {
  }


  virtual void parseValues()
  {
    m_sShape = SourceShape::fullDomain;

    setReleaseType();
    setParticleType();
    //Create particle factories
    registerParticles();
    /*
    // Create a generic particle with attributes read from XML
    Particles * particles;
    particles->setParticleValues();
*/
    //std::cout << " protoParticle->tag = " << protoParticle->tag << std::endl;


    parsePrimitive<double>(false, sourceStrength, "sourceStrength");
  }


  void checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend);


  int emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles);
};
