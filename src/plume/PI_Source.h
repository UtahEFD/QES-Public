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

/** @file Sources.hpp
* @brief This class contains data and variables that set flags and
* settngs read from the xml.
*
* @note Child of ParseInterface
* @sa ParseInterface
*/

#pragma once

#include "util/ParseInterface.h"

#include "SourceGeometry.hpp"
#include "SourceGeometry_Cube.hpp"
#include "SourceGeometry_FullDomain.hpp"
#include "SourceGeometry_Line.hpp"
#include "SourceGeometry_Point.hpp"
#include "SourceGeometry_SphereShell.hpp"

#include "ReleaseType.hpp"
#include "ReleaseType_instantaneous.hpp"
#include "ReleaseType_continuous.hpp"
#include "ReleaseType_duration.hpp"

class PI_Source : public ParseInterface
{
private:
protected:
  // ParseParticle *m_protoParticle{};
  SourceGeometry *m_sourceGeometry{};
  ReleaseType *m_releaseType{};

public:
  // constructor
  PI_Source() = default;

  // destructor
  virtual ~PI_Source() = default;

  int getNumParticles()
  {
    return m_releaseType->m_numPar;
  }

  void setReleaseType();
  void setSourceGeometry();
  void setParticleType();

  // accessor to geometry type
  SourceShape geometryType()
  {
    return m_sourceGeometry->m_sGeom;
  }
  // accessor to release type
  ParticleReleaseType releaseType()
  {
    return m_releaseType->parReleaseType;
  }

  void parseValues() override
  {
    setReleaseType();
    setParticleType();
    setSourceGeometry();
  }

  void checkPosInfo(const double &domainXstart,
                    const double &domainXend,
                    const double &domainYstart,
                    const double &domainYend,
                    const double &domainZstart,
                    const double &domainZend);

  void checkReleaseInfo(const double &timestep, const double &simDur);
  friend class Source;
};
