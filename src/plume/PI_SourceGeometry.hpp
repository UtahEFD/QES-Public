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

/** @file SourceType.hpp
 * @brief  This class represents a generic sourece type
 *
 * @note Pure virtual child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include <random>
#include <list>

// #include "Interp.h"

#include "util/ParseInterface.h"
#include "util/VectorMath.h"
// #include "winds/WINDSGeneralData.h"

enum SourceShape {
  point,
  line,
  sphereShell,
  cube,
  fullDomain
};

class PI_SourceGeometry : public ParseInterface
{
private:
  PI_SourceGeometry() = default;

protected:
  // ParticleTypeFactory *m_particleTypeFactory = nullptr;
  // ParseParticle *m_protoParticle = nullptr;

public:
  // this is a description variable for determining the source shape. May or may not be used.
  // !!! this needs set by parseValues() in each source generated from input files.
  SourceShape m_sGeom;

  // constructor
  explicit PI_SourceGeometry(const SourceShape &type) : m_sGeom(type)
  {
  }

  // destructor
  virtual ~PI_SourceGeometry() = default;


  // this function is used to parse all the variables for each source from the input .xml file
  // each source overloads this function with their own version, allowing different combinations of input variables for each source,
  // all these differences handled by parseInterface().
  // The = 0 at the end should force each inheriting class to require their own version of this function
  virtual void parseValues() = 0;

  // this function is for checking the source metadata to make sure all particles will be released within the domain.
  // There is one source so far (SourceFullDomain) that actually uses this function to set a few metaData variables
  //  specific to that source as well as to do checks to make sure particles stay within the domain.
  virtual void checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend) = 0;

  // this function set the initial position of each particle
  virtual void setInitialPosition(vec3 &) = 0;
};
