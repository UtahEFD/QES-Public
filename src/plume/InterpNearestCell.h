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

/** @file InterpTriLinear.h 
 * @brief 
 */

#pragma once

#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "util/calcTime.h"
#include "Random.h"
#include "util/Vector3.h"

#include "Interp.h"

class InterpNearestCell : public Interp
{

public:
  // constructor
  // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the InterpTriLinear grid values,
  // then calculates the tau gradients which are then used to calculate the flux_div grid values.
  InterpNearestCell(WINDSGeneralData *, TURBGeneralData *, const bool &);

  //double vel_threshold;

  void interpValues(const double &xPos,
                    const double &yPos,
                    const double &zPos,
                    const WINDSGeneralData *WGD,
                    double &uMain_out,
                    double &vMean_out,
                    double &wMean_out,
                    const TURBGeneralData *TGD,
                    double &txx_out,
                    double &txy_out,
                    double &txz_out,
                    double &tyy_out,
                    double &tyz_out,
                    double &tzz_out,
                    double &flux_div_x_out,
                    double &flux_div_y_out,
                    double &flux_div_z_out,
                    double &CoEps_out);

  void interpInitialValues(const double &xPos,
                           const double &yPos,
                           const double &zPos,
                           const TURBGeneralData *TGD,
                           double &sig_x_out,
                           double &sig_y_out,
                           double &sig_z_out,
                           double &txx_out,
                           double &txy_out,
                           double &txz_out,
                           double &tyy_out,
                           double &tyz_out,
                           double &tzz_out);

private:
  // timer class useful for debugging and timing different operations
  calcTime timers;

  // copies of debug related information from the input arguments
  bool debug;
};
