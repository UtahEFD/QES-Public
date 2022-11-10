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

/** @file ReleaseType.hpp 
 * @brief This class represents a generic particle release type. The idea is to make other classes that inherit from this class
 *  that are the specific release types, that make it easy to set the desired particle information for a given release type
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


class ReleaseType : public ParseInterface
{
protected:
public:
  // this is a description variable for determining the source release type. May or may not be used.
  // !!! this needs set by parseValues() in each source generated from input files.
  ParticleReleaseType parReleaseType;

  // LA-future work: might need to add another variable for the total number of available particles,
  //  or have a checking function that compares numParticles with totalNumParticles.

  int m_parPerTimestep;// this is the number of particles a given source needs to release each timestep
  double m_releaseStartTime;// this is the time a given source should start releasing particles
  double m_releaseEndTime;// this is the time a given source should end releasing particles
  int m_numPar;// this is the total number of particles expected to be released by a given source over the entire simulation


  // default constructor
  ReleaseType()
  {
  }

  // destructor
  virtual ~ReleaseType()
  {
  }


  // this function is used to parse all the variables for each release type in a given source from the input .xml file
  // each release type overloads this function with their own version, allowing different combinations of input variables for each release type,
  // all these differences handled by parseInterface().
  // The = 0 at the end should force each inheriting class to require their own version of this function
  // !!! in order for all the different combinations of input variables to work properly for each source, this function requires
  //  manually setting the variable parReleaseType in each version found in release types that inherit from this class.
  //  This is in addition to any other variables required for an individual release type that inherits from this class.
  virtual void parseValues() = 0;


  // this function is for setting the required inherited variables int m_parPerTimestep, double m_releaseStartTime,
  //  double m_releaseEndTime, and m_numPar. The way this is done differs for each release type inheriting from this class.
  // Note that this is a pure virtual function - enforces that the derived class MUST define this function
  //  this is done by the = 0 at the end of the function.
  // !!! Care must be taken to set all these variables in each inherited version of this function, where the calculated values
  //  will be able to pass the call to checkReleaseInfo() by the class setting up a vector of all the sources.
  // !!! each release type needs to have this function manually called for them by whatever class sets up a vector of this class.
  // !!! LA note and warn: the numPar is a bit tricky to calculate correctly because the nReleaseTimes is tough to calculate correctly
  //  if the simulation list of times has a major change, this may lead to a change in the way numPar is calculated.
  //  For now, the method is as follows: if you have times 0 to 10 with timestep 1, the nTimes should be 11
  //  BUT releasing 10 particles at each of the times up to nTimes would result in 110 particles. At the same time, the
  //  simultion time loop goes from 0 to nTimes-2 NOT 0 to nTimes-1 BECAUSE each time iteration is actually calculating particles positions
  //  for the next time, not the current time. THIS MEANS particles should be released over nTimes-1, NOT over nTimes like you would think.
  //  For both a list of times from 0 to 10 with timesteps 1 and 4, std::ceil(releaseDur/dt) = nTimes-1 NOT nTimes, which is EXACTLy what we want.
  //  !!! Care should be taken to use nReleaseTimes = std::ceil(releaseDur/dt) each instance to get the correct number of times particles should be released.
  virtual void calcReleaseInfo(const double &timestep, const double &simDur) = 0;


  // this function is for checking the set release type variables to make sure they are consistent with simulation information.
  // !!! each release type needs to have this function manually called for them by whatever class sets up a vector of this class.
  // LA-note: the check functions are starting to be more diverse and in different spots.
  //  Maybe a better name for this function would be something like checkReleaseTypeInfo().
  // LA-warn: should this be virtual? The idea is that I want it to stay as this function no matter what ReleaseType is chosen,
  //  I don't want this function overloaded by any classes inheriting this class.
  // !!! LA note and warn: the numPar is a bit tricky to calculate correctly because the nReleaseTimes is tough to calculate correctly
  //  if the simulation list of times has a major change, this may lead to a change in the way numPar is calculated.
  //  For now, the method is as follows: if you have times 0 to 10 with timestep 1, the nTimes should be 11
  //  BUT releasing 10 particles at each of the times up to nTimes would result in 110 particles. At the same time, the
  //  simultion time loop goes from 0 to nTimes-2 NOT 0 to nTimes-1 BECAUSE each time iteration is actually calculating particles positions
  //  for the next time, not the current time. THIS MEANS particles should be released over nTimes-1, NOT over nTimes like you would think.
  //  For both a list of times from 0 to 10 with timesteps 1 and 4, std::ceil(releaseDur/dt) = nTimes-1 NOT nTimes, which is EXACTLy what we want.
  //  !!! Care should be taken to use nReleaseTimes = std::ceil(releaseDur/dt) each instance to get the correct number of times particles should be released.
  virtual void checkReleaseInfo(const double &timestep, const double &simDur)
  {
    if (m_parPerTimestep <= 0) {
      std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input m_parPerTimestep is <= 0!";
      std::cerr << " m_parPerTimestep = \"" << m_parPerTimestep << "\"" << std::endl;
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
      if (m_numPar != m_parPerTimestep) {
        std::cerr << "ERROR (ReleaseType::checkReleaseInfo): input ParticleReleaseType is instantaneous but input m_numPar does not equal input m_parPerTimestep!";
        std::cerr << " m_numPar = \"" << m_numPar << "\", m_parPerTimestep = \"" << m_parPerTimestep << "\"" << std::endl;
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
      if (m_parPerTimestep * nReleaseTimes != m_numPar) {
        std::cerr << "ERROR (ReleaseType::checkReleaseInfo): calculated particles for release does not match input m_numPar!";
        std::cerr << " m_parPerTimestep = \"" << m_parPerTimestep << "\", nReleaseTimes = \"" << nReleaseTimes
                  << "\", m_numPar = \"" << m_numPar << "\"" << std::endl;
        exit(1);
      }
    }
  }
};
