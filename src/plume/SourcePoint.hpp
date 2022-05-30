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

/** @file SourcePoint.hpp
 * @brief This class represents a specific source type. 
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once


#include "SourceType.hpp"
#include "winds/WINDSGeneralData.h"
//#include "Particles.hpp"

class SourcePoint : public SourceType
{
private:
  // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
  // guidelines for how to set these variables within an inherited source are given in SourceType.

  double posX;
  double posY;
  double posZ;
  //  double pRho = 0.0; // particle density (kg/m^3), variable read in from XML // LDU commented 11/16
  //  double pD = 0.0; // particle diameter (microns), variable read in from XML // LDU commented 11/16
  double sourceStrength = 0.0;// total mass released (g)
  //  bool sourceDepFlag = true; // deposition flag (1 for on, 0 for off)
protected:
public:
  // Default constructor
  SourcePoint()
  {
  }

  // destructor
  ~SourcePoint()
  {
  }


  virtual void parseValues()
  {
    m_sShape = SourceShape::point;

    setReleaseType();
    setParticleType();
    // Create particle factories
    registerParticles();

    /* 
    // Create a generic particle with attributes read from XML
    Particles * particles;
    particles->setParticleValues();
*/
    //std::cout << " protoParticle->tag = " << protoParticle->tag << std::endl;
    parsePrimitive<double>(true, posX, "posX");
    parsePrimitive<double>(true, posY, "posY");
    parsePrimitive<double>(true, posZ, "posZ");

    //    parsePrimitive<double>(false, pRho, "particleDensity"); // LDU commented 11/16
    //    parsePrimitive<double>(false, pD, "particleDiameter"); // LDU commented 11/16
    parsePrimitive<double>(false, sourceStrength, "sourceStrength");
    //    parsePrimitive<bool>(false, sourceDepFlag, "depositionFlag"); // LDU commented 11/16
  }


  void checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend);

  //template <class parType>
  int emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles, WINDSGeneralData *WGD);
};
