/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file FIREOutput.h
 * @brief This class handles saving output files for Fire related variables
 * This is a specialized output class derived and inheriting from QESNetCDFOutput.
 */
#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include "winds/WINDSGeneralData.h"
#include "winds/WINDSInputData.h"
#include "util/QESNetCDFOutput.h"
#include "util/QEStime.h"
#include "Fire.h"

/* Specialized output classes derived from QESNetCDFOutput for 
   cell center data (used primarly for vizualization)
*/
class FIREOutput : public QESNetCDFOutput
{
public:
  FIREOutput(WINDSGeneralData *, Fire *, std::string);
  ~FIREOutput()
  {}

  void save(QEStime);

private:
  // data container for output (on cell-center without ghost cell)
  std::vector<float> x_out, y_out, z_out;
  std::vector<int> icellflag_out;
  std::vector<float> u_out, v_out, w_out;

  // copy of pointer for data access
  WINDSGeneralData *wgd_;
  Fire *fire_;
};
