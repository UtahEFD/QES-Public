/****************************************************************************
 * Copyright (c) 2024 University of Utah
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
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

#include "Fire.h"
/**
 * @file Rothermel.cpp
 *
 * @brief This function calculates the ROS according to the Rothermel model
 *  
 */
float Fire ::rothermel(FuelProperties *fuel, float max_wind, float tanphi, float fmc_g)
{

  // fuel properties
  float oneHour = fuel->oneHour;///< one hour fuel load [t/ac]
  float tenHour = fuel->tenHour;///< ten hour fuel load [t/ac]
  float hundredHour = fuel->hundredHour;///< hundred hour fuel load [t/ac]
  float liveHerb = fuel->liveHerb;///< live herb fuel load [t/ac]
  float liveWoody = fuel->liveWoody;///< live woody fuel load [t/ac]
  float savOneHour = fuel->savOneHour;///< surface area to volume ratio of one hour fuel load
  float savTenHour = 109;///< surface area to volume ratio of ten hour fuel load
  float savHundredHour = 30;///< surface area to volume ratio of hundred hour fuel load
  float savHerb = fuel->savHerb;///< surface area to volume ratio of herbacious fuel load
  float savWoody = fuel->savWoody;///< surface area to volume ratio of live woody fuel load
  float fueldens = fuel->fuelDensity;///< ovendry fuel particle density [lb/ft^2]
  float st = 0.0555;///< Total fuel mineral content
  float se = 0.0100;///< Silica free mineral content
  float fgi = (oneHour + liveHerb + liveWoody) * 0.2471;///< Initial fine fuel load [kg/m^2]
  float fuelmce = fuel->fuelmce / 100;///< fuel moisture content of extinction[%]
  float fuelheat = fuel->heatContent;///< heat content of fuel [BTU/lb]
  float fueldepth = fuel->fuelDepth;///< fuel bed depth [ft]

  float savr = (savOneHour * oneHour + savTenHour * tenHour + savHundredHour * hundredHour + savHerb * liveHerb + savWoody * liveWoody)
               / (oneHour + tenHour + hundredHour + liveHerb + liveWoody);///< Characteristic fine fuel load surface area to volume ratio
  // local fire variables
  float bmst = fmc_g / (1 + fmc_g);
  float fuelloadm = (1. - bmst) * fgi;
  float fuelload = fuelloadm * (pow(.3048, 2.0)) * 2.205;// convert fuel load to lb/ft^2
  float betafl = fuelload / (fueldepth * fueldens);// packing ratio  jm: lb/ft^2/(ft * lb*ft^3) = 1
  float betaop = 3.348 * pow(savr, -0.8189);// optimum packing ratio jm: units??
  float qig = 250. + 1116. * fmc_g;// heat of preignition, btu/lb
  float epsilon = exp(-138. / savr);// effective heating number
  float rhob = fuelload / fueldepth;// ovendry bulk density, lb/ft^3
  float rtemp2 = pow(savr, 1.5);
  float gammax = rtemp2 / (495. + 0.0594 * rtemp2);// maximum rxn vel, 1/min
  float ar = 1. / (4.774 * (pow(savr, 0.1)) - 7.27);// coef for optimum rxn vel
  float ratio = betafl / betaop;
  float gamma = gammax * ((pow(ratio, ar)) * (exp(ar * (1. - ratio))));// optimum rxn vel, 1/min
  float wn = fuelload / (1 + st);// net fuel loading, lb/ft^2
  float rtemp1 = fmc_g / fuelmce;
  float etam = 1. - 2.59 * rtemp1 + 5.11 * (pow(rtemp1, 2)) - 3.52 * (pow(rtemp1, 3));// moist damp coef
  float etas = 0.174 * pow(se, -0.19);// mineral damping coef
  float ir = gamma * wn * fuelheat * etam * etas;// rxn intensity,btu/ft^2 min
  float xifr = exp((0.792 + 0.681 * (pow(savr, 0.5))) * (betafl + 0.1)) / (192. + 0.2595 * savr);// propagating flux ratio
  float rothR0 = ir * xifr / (rhob * epsilon * qig);// SPREAD RATE [ft/s]
  float R0 = rothR0 * .005080;// SPREAD RATE [m/s]

  return R0;
}