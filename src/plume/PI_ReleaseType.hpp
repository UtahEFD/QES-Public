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

/** @file PI_ReleaseType.hpp
 * @brief This class represents a generic particle release type.
 * The idea is to make other classes that inherit from this class
 * that are the specific release types, that make it easy to set the desired particle
 * information for a given release type
 *
 * @note Pure virtual child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include <cmath>

#include "util/ParseInterface.h"

#include "SourceReleaseController.h"

enum ParticleReleaseType {
  instantaneous,
  continuous,
  duration
};

class PI_ReleaseType : public ParseInterface,
                       public SourceReleaseControllerBuilderInterface
{
private:
protected:
public:// [FM] to remove
  // default constructor
  PI_ReleaseType() = default;

  // destructor
  virtual ~PI_ReleaseType() = default;

  // Number of particles to release each timestep
  int m_particlePerTimestep = -1;
  // Mass to release each timestep
  float m_massPerTimestep = 0.0;
  // Time the source starts releasing particles
  float m_releaseStartTime = -1.0;
  // Time the source ends releasing particles
  float m_releaseEndTime = -1.0;

public:
  /**
   * /brief This function is used to parse all the variables for each release type
   * in a given source from the input .xml file each release type overloads this function
   * with their own version, allowing different combinations of input variables for each release type,
   * all these differences handled by parseInterface().
   */
  virtual void parseValues() = 0;

  /**
   * /brief This function is for setting the required inherited variables int m_particlePerTimestep,
   * float m_releaseStartTime, float m_releaseEndTime, and m_numPar
   *
   * @param timestep   simulation time step.
   * @param simDur     simulation duration.
   */
  virtual void initialize(const float &timestep) = 0;
};
