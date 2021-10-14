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

/** @file PlumeOutputEulerian.h
 * @brief This class handles saving output files for Eulerian data.
 * This is a specialized output class derived and inheriting from QESNetCDFOutput.
 *
 * @note child of QESNetCDFOutput
 * @sa QESNetCDFOutput
 */

#pragma once

#include <string>

#include "PlumeInputData.hpp"
#include "src/winds/WINDSGeneralData.h"
#include "src/winds/TURBGeneralData.h"
#include "Eulerian.h"

#include "util/QESNetCDFOutput.h"

class PlumeOutputEulerian : public QESNetCDFOutput
{
public:
  // default constructor
  PlumeOutputEulerian() : QESNetCDFOutput()
  {
  }

  // specialized constructor
  PlumeOutputEulerian(PlumeInputData *, WINDSGeneralData *, TURBGeneralData *, Eulerian *, std::string);

  // deconstructor
  ~PlumeOutputEulerian()
  {
  }

  // setup and save output for the given time
  // in this case the saved data is output averaged concentration
  // This is the one function that needs called from outside after constructor time
  void save(float currentTime);
  void save(ptime) {}

private:
  // no need for output frequency for this output, it is expected to only happen once, assumed to be at time zero

  // pointers to the classes that save needs to use to get the data for the output
  WINDSGeneralData *WGD_;
  TURBGeneralData *TGD_;
  Eulerian *eul_;

  // other output data
  std::vector<float> epps;// data is normally stored as CoEps, so need to separate it out here
};
