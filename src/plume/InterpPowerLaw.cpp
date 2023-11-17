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

/** @file interpPowerLaw.cpp */

#include "InterpPowerLaw.h"


InterpPowerLaw::InterpPowerLaw(WINDSGeneralData *WGD, TURBGeneralData *TGD, const bool &debug_val)
  : Interp(WGD)
{
  // std::cout << "[InterpPowerLaw] \t Setting Interpolation method " << std::endl;

  // copy debug information
  bool debug = debug_val;

  if (debug == true) {
    std::cout << "[InterpPowerLaw] \t DEBUG - Domain boundary" << std::endl;
    std::cout << "\t\t xStart=" << xStart << " xEnd=" << xEnd << std::endl;
    std::cout << "\t\t yStart=" << yStart << " yEnd=" << yEnd << std::endl;
    std::cout << "\t\t zStart=" << zStart << " zEnd=" << zEnd << std::endl;
  }
}

double InterpPowerLaw::getMaxFluctuation()
{
  // calculate the threshold velocity
  double a = 4.8;
  double p = 0.15;

  double us = 0.4 * p * a * pow(zEnd, p);
  return 10.0 * 2.5 * us;
}


void InterpPowerLaw::interpInitialValues(const double &xPos,
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
                                         double &tzz_out)
{
  double a = 4.8;
  double p = 0.15;

  // double b = 0.08;
  // double n = 1.0;
  double z = 0.1 * ceil(10.0 * zPos);

  double us = 0.4 * p * a * pow(z, p);
  // double us = sqrt(b * p * a * pow(zPos, n + p - 1));

  sig_x_out = 2.5 * us;
  sig_y_out = 2.3 * us;
  sig_z_out = 1.3 * us;

  txx_out = pow(sig_x_out, 2.0);
  tyy_out = pow(sig_y_out, 2.0);
  tzz_out = pow(sig_z_out, 2.0);
  txy_out = 0.0;
  tyz_out = 0.0;
  txz_out = -pow(us, 2.0);

  return;
}

void InterpPowerLaw::interpValues(const WINDSGeneralData *WGD,
                                  const double &xPos,
                                  const double &yPos,
                                  const double &zPos,
                                  double &uMean_out,
                                  double &vMean_out,
                                  double &wMean_out)
{
  double a = 4.8;
  double p = 0.15;

  // double b = 0.08;
  // double n = 1.0;

  double z = 0.1 * ceil(10.0 * zPos);

  double us = 0.4 * p * a * pow(z, p);
  // double us = sqrt(b * p * a * pow(z, n + p - 1));

  uMean_out = a * pow(z, p);
  vMean_out = 0.0;
  wMean_out = 0.0;
}

void InterpPowerLaw::interpValues(const TURBGeneralData *TGD,
                                  const double &xPos,
                                  const double &yPos,
                                  const double &zPos,
                                  double &txx_out,
                                  double &txy_out,
                                  double &txz_out,
                                  double &tyy_out,
                                  double &tyz_out,
                                  double &tzz_out,
                                  double &flux_div_x_out,
                                  double &flux_div_y_out,
                                  double &flux_div_z_out,
                                  double &nuT_out,
                                  double &CoEps_out)
{
  double a = 4.8;
  double p = 0.15;

  // double b = 0.08;
  // double n = 1.0;

  double z = 0.1 * ceil(10.0 * zPos);

  double us = 0.4 * p * a * pow(z, p);
  // double us = sqrt(b * p * a * pow(z, n + p - 1));


  CoEps_out = (5.7 * us * us * us) / (0.4 * z);

  txx_out = pow(2.5 * us, 2.0);
  tyy_out = pow(2.3 * us, 2.0);
  tzz_out = pow(1.3 * us, 2.0);
  txy_out = 0.0;
  tyz_out = 0.0;
  txz_out = -pow(us, 2.0);

  flux_div_x_out = -2.0 * p * pow(0.4 * p * a, 2.0) * pow(z, 2.0 * p - 1.0);
  // flux_div_x_out = -b * p * a * (n + p - 1) * pow(z, n + p - 2.0);
  flux_div_y_out = 0.0;
  flux_div_z_out = 2.0 * p * pow(1.3 * 0.4 * p * a, 2.0) * pow(z, 2.0 * p - 1.0);
  // flux_div_z_out = (1.3 * 1.3) * b * p * a * (n + p - 1) * pow(z, n + p - 2.0);
}
