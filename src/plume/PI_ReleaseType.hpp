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

enum ParticleReleaseType {
  instantaneous,
  continuous,
  duration
};

class PI_ReleaseType : public ParseInterface
{
private:
  // default constructor
  PI_ReleaseType() = default;

protected:
  // Total number of particles expected to be released by the source over
  // the entire simulation
  int m_numPar = -1;
  // Total mass to be released [g]
  double m_totalMass = 0.0;
  // Mass [g/s] to be released
  double m_massPerSec = 0.0;

public:
  // Description variable for source release type.
  // set by the constructor of the derived classes.
  ParticleReleaseType parReleaseType{};

  // Number of particles to release each timestep
  int m_particlePerTimestep = -1;
  // Time the source starts releasing particles
  double m_releaseStartTime = -1.0;
  // Time the source ends releasing particles
  double m_releaseEndTime = -1.0;
  // Mass per particle
  double m_massPerParticle = 0.0;

  explicit PI_ReleaseType(const ParticleReleaseType &type)
    : parReleaseType(type)
  {
  }

  // destructor
  virtual ~PI_ReleaseType() = default;


  /**
   * /brief This function is used to parse all the variables for each release type
   * in a given source from the input .xml file each release type overloads this function
   * with their own version, allowing different combinations of input variables for each release type,
   * all these differences handled by parseInterface().
   */
  virtual void parseValues() = 0;

  /**
   * /brief This function is for setting the required inherited variables int m_particlePerTimestep,
   * double m_releaseStartTime, double m_releaseEndTime, and m_numPar
   *
   * @param timestep   simulation time step.
   * @param simDur     simulation duration.
   */
  virtual void calcReleaseInfo(const double &timestep, const double &simDur) = 0;

  /**
   * /brief This function is for checking the set release type variables to make sure
   * they are consistent with simulation information.
   *
   * @param timestep   simulation time step.
   * @param simDur     simulation duration.
   */
  virtual void checkReleaseInfo(const double &timestep, const double &simDur)
  {
    if (m_particlePerTimestep <= 0) {
      std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_particlePerTimestep is <= 0!";
      std::cerr << " m_particlePerTimestep = \"" << m_particlePerTimestep << "\"" << std::endl;
      exit(1);
    }
    if (m_releaseStartTime < 0) {
      std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_releaseStartTime is < 0!";
      std::cerr << " m_releaseStartTime = \"" << m_releaseStartTime << "\"" << std::endl;
      exit(1);
    }
    if (m_releaseEndTime > simDur) {
      std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_releaseEndTime is > input simDur!";
      std::cerr << " m_releaseEndTime = \"" << m_releaseEndTime << "\", simDur = \"" << simDur << "\"" << std::endl;
      exit(1);
    }
    if (m_releaseEndTime < m_releaseStartTime) {
      std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_releaseEndTime is < input m_releaseStartTime!";
      std::cerr << " m_releaseStartTime = \"" << m_releaseStartTime << "\", m_releaseEndTime = \"" << m_releaseEndTime << "\"" << std::endl;
      exit(1);
    }

    // this one is a bit trickier to check. Specifically the way the number of timesteps for a given release
    //  is calculated needs to be watched carefully to make sure it is consistent throughout the entire program
    double releaseDur = m_releaseEndTime - m_releaseStartTime;
    if (parReleaseType == ParticleReleaseType::instantaneous) {
      if (releaseDur != 0) {
        std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is instantaneous but input m_releaseStartTime does not equal m_releaseEndTime!";
        std::cerr << " m_releaseStartTime = \"" << m_releaseStartTime << "\", m_releaseEndTime = \"" << m_releaseEndTime << "\"" << std::endl;
        exit(1);
      }
      if (m_numPar != m_particlePerTimestep) {
        std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is instantaneous but input m_numPar does not equal input m_particlePerTimestep!";
        std::cerr << " m_numPar = \"" << m_numPar << "\", m_particlePerTimestep = \"" << m_particlePerTimestep << "\"" << std::endl;
        exit(1);
      }
    } else {
      // Again, the way the number of timesteps for a given release
      //  is calculated needs to be watched carefully to make sure it is consistent throughout the program
      int nReleaseTimes = std::ceil(releaseDur / timestep);
      if (nReleaseTimes == 0) {
        std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is not instantaneous but calculated nReleaseTimes is zero!";
        std::cerr << " nReleaseTimes = \"" << nReleaseTimes << "\", releaseDur = \"" << releaseDur
                  << "\", timestep = \"" << timestep << "\"" << std::endl;
        exit(1);
      }
      if (m_particlePerTimestep * nReleaseTimes != m_numPar) {
        std::cerr << "ERROR (ReleaseType::checkReleaseInfo): calculated particles for release does not match input m_numPar!";
        std::cerr << " m_particlePerTimestep = \"" << m_particlePerTimestep << "\", nReleaseTimes = \"" << nReleaseTimes
                  << "\", m_numPar = \"" << m_numPar << "\"" << std::endl;
        exit(1);
      }
    }
  }
};
