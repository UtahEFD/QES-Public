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

/** @file PlumeParameters.hpp 
 * @brief This class contains data and variables that set flags and
 * settngs read from the xml.
 *
 * @note Child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include "util/ParseInterface.h"
#include <string>
#include <cmath>

class PlumeParameters : public ParseInterface
{

private:
public:
  float simDur; /**< amount of time to run the simulation */
  float timeStep; /**< overall integration timestep */
  float CourantNum; /**< Courant Number,
		       how to divide the timestep up for each particle to keep them 
		       moving one grid cell at a time */
  double invarianceTol; /** < tolerance used to determine whether makeRealizeable should be run 
			   on the stress tensor for a particle */

  int updateFrequency_particleLoop; /**< frequency to print out information during the particle loop of the solver. 
				      Only used during debug mode */
  float updateFrequency_timeLoop; /**< frequency to print out information during the time integration loop of the solver. */

  std::string interpMethod; /**< interpolation method:
			       triLinear - tri linear interpolation (default)
			       nearestCell - use value from the cell where the particle is 
			       analyticalPowerLaw - use analytical solution from the power law
			    */

  /**
   * Parse the input file for parameters.
   */
  virtual void parseValues()
  {
    parsePrimitive<float>(true, simDur, "simDur");
    parsePrimitive<float>(true, timeStep, "timeStep");

    parsePrimitive<float>(true, CourantNum, "CourantNumber");
    parsePrimitive<double>(true, invarianceTol, "invarianceTol");

    parsePrimitive<int>(true, updateFrequency_particleLoop, "updateFrequency_particleLoop");
    parsePrimitive<float>(true, updateFrequency_timeLoop, "updateFrequency_timeLoop");

    interpMethod = "triLinear";
    parsePrimitive<std::string>(false, interpMethod, "interpolationMethod");

    // check some of the parsed values to see if they make sense
    checkParsedValues();
  }

  /**
   * Saves user-defined data to file.
   */
  void checkParsedValues()
  {
    // make sure simDur, timeStep, and invarianceTol are not negative values
    if (simDur <= 0) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input simDur must be greater than zero!";
      std::cerr << " simDur = \"" << simDur << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (timeStep <= 0) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input timeStep must be greater than zero!";
      std::cerr << " timeStep = \"" << timeStep << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (CourantNum < 0.0 || CourantNum > 1.0) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input CourantNumber must be greater than or equal to zero but less than or equal to one!";
      std::cerr << " CourantNumber = \"" << CourantNum << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (invarianceTol <= 0) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input invarianceTol must be greater than zero!";
      std::cerr << " invarianceTol = \"" << invarianceTol << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }

    // make sure the updateFrequency variables are value 1 or greater
    if (updateFrequency_particleLoop < 1) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input updateFrequency_particleLoop must be 1 or greater!";
      std::cerr << " updateFrequency_particleLoop = \"" << updateFrequency_particleLoop << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (updateFrequency_timeLoop < 0) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input updateFrequency_timeLoop must be 1 or greater!";
      std::cerr << " updateFrequency_timeLoop = \"" << updateFrequency_timeLoop << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }


    // make sure the input timestep is not greater than the simDur
    if (timeStep > simDur) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input timeStep must be smaller than or equal to input simDur!";
      std::cerr << " timeStep = \"" << timeStep << "\", simDur = \"" << simDur << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }


    // make sure the updateFrequency time loop variable is not bigger than the simulation duration
    // LA note: this is not the same way nTimes is calculated in plume, I'm assuming that zero doesn't matter
    int nTimes = std::ceil(simDur / timeStep);
    if (updateFrequency_timeLoop > nTimes) {
      std::cerr << "(SimulationParameters::checkParsedValues): "
                << "input updateFrequency_timeLoop must be smaller than or equal to calculated nTimes!";
      std::cerr << " updateFrequency_timeLoop = \"" << updateFrequency_timeLoop << "\", nTimes = \"" << nTimes << "\"" << std::endl;
    }
  }
};
