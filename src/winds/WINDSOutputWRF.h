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
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

#pragma once

#include <string>

#include "WINDSGeneralData.h"
#include "WRFInput.h"
#include "util/QESNetCDFOutput.h"

/* Specialized output classes derived from QESNetCDFOutput for
   face center data (used for turbulence,...)
*/
class WINDSOutputWRF : public QESNetCDFOutput
{
public:
  WINDSOutputWRF()
    : QESNetCDFOutput()
  {}

  WINDSOutputWRF(WINDSGeneralData *, WRFInput *wrfInputData);

  ~WINDSOutputWRF() {}

  // save function be call outside
  void save(ptime);

private:
  WINDSGeneralData *WGD_;
  WRFInput *wrf_;
};
