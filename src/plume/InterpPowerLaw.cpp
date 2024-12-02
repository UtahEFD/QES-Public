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

/** @file interpPowerLaw.cpp */

#include "InterpPowerLaw.h"


InterpPowerLaw::InterpPowerLaw(qes::Domain domain_in, bool debug_val = 0)
  : Interp(domain_in)
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

void InterpPowerLaw::interpWindsValues(const WINDSGeneralData *WGD,
                                       const vec3 &pos,
                                       vec3 &vel_out)
{
  float a = 4.8;
  float p = 0.15;

  // double b = 0.08;
  // double n = 1.0;

  float z = 0.1 * ceil(10.0 * pos._3);

  float us = 0.4 * p * a * powf(z, p);
  // double us = sqrt(b * p * a * pow(z, n + p - 1));

  vel_out._1 = a * powf(z, p);
  vel_out._2 = 0.0;
  vel_out._3 = 0.0;
}

void InterpPowerLaw::interpTurbValues(const TURBGeneralData *TGD,
                                      const vec3 &pos,
                                      mat3sym &tau_out,
                                      vec3 &flux_div_out,
                                      float &nuT_out,
                                      float &CoEps_out)
{
  float a = 4.8;
  float p = 0.15;

  // double b = 0.08;
  // double n = 1.0;

  float z = 0.1 * ceil(10.0 * pos._3);

  float us = 0.4 * p * a * pow(z, p);
  // double us = sqrt(b * p * a * pow(z, n + p - 1));

  CoEps_out = (5.7f * us * us * us) / (0.4f * z);

  tau_out = { powf(2.5f * us, 2.0f), powf(2.3f * us, 2.0f), powf(1.3f * us, 2.0f), 0.0f, 0.0f, -powf(us, 2.0f) };

  flux_div_out._1 = -2.0f * p * powf(0.4f * p * a, 2.0f) * powf(z, 2.0f * p - 1.0f);
  // flux_div_out.1 = -b * p * a * (n + p - 1) * pow(z, n + p - 2.0);
  flux_div_out._2 = 0.0f;
  flux_div_out._3 = 2.0f * p * powf(1.3f * 0.4f * p * a, 2.0f) * powf(z, 2.0f * p - 1.0f);
  // flux_div_out._3 = (1.3 * 1.3) * b * p * a * (n + p - 1) * pow(z, n + p - 2.0);
}


void InterpPowerLaw::interpTurbInitialValues(const TURBGeneralData *TGD,
                                             const vec3 &pos,
                                             mat3sym &tau_out,
                                             vec3 &sig_out)
{
  float a = 4.8;
  float p = 0.15;

  // double b = 0.08;
  // double n = 1.0;
  float z = 0.1 * ceil(10.0 * pos._3);

  float us = 0.4 * p * a * powf(z, p);
  // double us = sqrt(b * p * a * pow(zPos, n + p - 1));

  tau_out = { powf(2.5f * us, 2.0f), powf(2.3f * us, 2.0f), powf(1.3f * us, 2.0f), 0.0f, 0.0f, -powf(us, 2.0f) };
  sig_out = { 2.5f * us, 2.3f * us, 1.3f * us };
}
