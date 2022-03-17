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

/** @file PlumeOutputParticleData.h
 * @brief This class handles saving output files for Lagrangian particle data.
 * This is a specialized output class derived and inheriting from QESNetCDFOutput.
 *
 * @note child of QESNetCDFOutput
 * @sa QESNetCDFOutput
 */

#pragma once

#include <string>

#include "PlumeInputData.hpp"

#include "util/QESNetCDFOutput.h"

class Plume;

class PlumeOutputParticleData : public QESNetCDFOutput
{
public:
  // default constructor
  PlumeOutputParticleData() : QESNetCDFOutput()
  {
  }

  // specialized constructor
  PlumeOutputParticleData(PlumeInputData *PID, Plume *plume_ptr, std::string output_file);

  // deconstructor
  ~PlumeOutputParticleData()
  {
  }

  // setup and save output for the given time
  // in this case the saved data is output averaged concentration
  // This is the one function that needs called from outside after constructor time
  void save(float);
  void save(ptime) {}

protected:
  bool validateFileOtions();

private:
  // Output frequency control information
  float outputStartTime;// time to start output, adjusted if the output duration does not divide evenly by the output frequency
  float outputEndTime;// time to end output
  float outputFrequency;// output frequency


  // variables needed for getting proper output time control
  float nextOutputTime;// next output time value that is updated each time save is called and there is output

  // pointer to the class that save needs to use to get the data for the concentration calculation
  Plume *plume;

  // all possible output fields need to be add to this list
  std::vector<std::string> allOutputFields = { "parID", "tStrt", "sourceIdx", "d", "m", "wdepos", "wdecay", "xPos_init", "yPos_init", "zPos_init", "xPos", "yPos", "zPos", "uMean", "vMean", "wMean", "uFluct", "vFluct", "wFluct", "delta_uFluct", "delta_vFluct", "delta_wFluct", "isRogue", "isActive" };
  std::vector<std::string> minimalOutputFields = { "parID", "tStrt", "sourceIdx", "xPos", "yPos", "zPos", "isActive" };


  // main particle metadata, almost a copy of what is in the "particle" class
  int numPar;// total number of particle to be released
  std::vector<int> parID;// list of particle IDs (for NetCDF dimension)
  std::vector<float> tStrt;// list of release times for the particles
  std::vector<int> sourceIdx;// list of sourceIdx for the particles

  std::vector<float> d;// list of particle diameter
  std::vector<float> m;// list of particle mass
  std::vector<float> wdepos;// list of particle non-deposited fraction
  std::vector<float> wdecay;// list of particle non-decay fraction

  std::vector<float> xPos_init;// list of initial x positions for the particles
  std::vector<float> yPos_init;// list of initial y positions for the particles
  std::vector<float> zPos_init;// list of initial z positions for the particles

  // other particle data, definitely a copy of what is in the "particle" class,
  // but only the particle information that matters for particle statistic calculations
  // and particle info plotting
  std::vector<float> xPos;// list of x positions for the particles
  std::vector<float> yPos;// list of y positions for the particles
  std::vector<float> zPos;// list of z positions for the particles
  std::vector<float> uMean;// list of u mean velocity for the particles
  std::vector<float> vMean;// list of v mean velocity for the particles
  std::vector<float> wMean;// list of w mean velocity for the particles
  std::vector<float> uFluct;// list of u velocity fluctuations for the particles
  std::vector<float> vFluct;// list of v velocity fluctuations for the particles
  std::vector<float> wFluct;// list of w velocity fluctuations for the particles
  std::vector<float> delta_uFluct;// list of the uFluct differences for the particles
  std::vector<float> delta_vFluct;// list of the vFluct differences for the particles
  std::vector<float> delta_wFluct;// list of the wFluct differences for the particles
  std::vector<int> isRogue;// list of isRogue info for the particles
  std::vector<int> isActive;// list of isActive info for the particles
};
