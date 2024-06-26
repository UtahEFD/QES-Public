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

/** @file PlumeOutput.h
 * @brief This class handles saving output files for Eulerian binned Lagrangian particle data,
 * where this class handles the binning of the Lagrangian particle data
 * This is a specialized output class derived and inheriting from QESNetCDFOutput.
 *
 * @note child of QESNetCDFOutput
 * @sa QESNetCDFOutput
 */

#pragma once

#include <string>

#include "PlumeInputData.hpp"
#include "winds/WINDSGeneralData.h"

#include "util/QESNetCDFOutput.h"
#include "util/QEStime.h"

class Plume;

class PlumeOutput : public QESNetCDFOutput
{
public:
  // specialized constructor
  PlumeOutput(const PlumeInputData *PID, Plume *plume_ptr, std::string output_file);

  // deconstructor
  ~PlumeOutput()
  {
  }

  // setup and save output for the given time
  // in this case the saved data is output averaged concentration
  // This is the one function that needs called from outside after constructor time
  void save(QEStime);

  // averaging period in seconds
  float averagingPeriod;
  // need the simulation timeStep for use in concentration averaging
  float timeStep;
  // Sampling box variables for calculating concentration data
  // Number of boxes to use for the sampling box
  int nBoxesX, nBoxesY, nBoxesZ;// Copies of the input: nBoxesX, Y, and Z.
  // upper & lower bounds in each direction of the sampling boxes
  float lBndx, lBndy, lBndz, uBndx, uBndy, uBndz;// Copies of the input: boxBoundsX1, boxBoundsX2, boxBoundsY1,
  float boxSizeX, boxSizeY, boxSizeZ;// these are the box sizes in each direction, calculated from nBoxes, lBnd, and uBnd variables
  float volume;// volume of the sampling boxes (=nBoxesX*nBoxesY*nBoxesZ)

  // output concentration storage variables
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<int> pBox;// sampling box particle counter (for average)
  std::vector<float> conc;// concentration values (for output)

private:
  // default constructor
  PlumeOutput() {}

  // time averaging frequency control information
  // in this case, this is also the output control information
  // time to start concentration averaging, not the time to start output.
  QEStime averagingStartTime;
  //
  float ongoingAveragingTime;

  // variables needed for getting proper averaging and output time control
  // next output time value that is updated each time save is called and output occurs.
  // Also initializes a restart of the particle binning for the next time averaging period
  QEStime nextOutputTime;

  // pointer to the class that save needs to use to get the data for the concentration calculation
  Plume *m_plume;


  // function for counting the number of particles in the sampling boxes
  void boxCount();
};
