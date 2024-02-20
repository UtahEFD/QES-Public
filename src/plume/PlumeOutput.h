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


class PlumeInputData;
class PLUMEGeneralData;

class PlumeOutput : public QESNetCDFOutput
{
public:
  // specialized constructor
  PlumeOutput(const PlumeInputData *PID, PLUMEGeneralData *PGD, const std::string &output_file);

  // destructor
  ~PlumeOutput()
  {
  }

  // setup and save output for the given time
  void save(QEStime) override;

private:
  // default constructor
  PlumeOutput() = default;

  // time averaging frequency control information
  // in this case, this is also the output control information
  // time to start concentration averaging.
  QEStime outputStartTime;

  // averaging period in seconds
  float outputPeriod{};

  // variables needed for getting proper output time control
  // next output time value that is updated each time save is called and output occurs.
  QEStime nextOutputTime;

  // pointer to the class that save needs to use to get the data for the concentration calculation
  PLUMEGeneralData *m_PGD{};
};
