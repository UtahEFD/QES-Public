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
 * @file Balbi.cpp
 * @brief This function calculates the ROS and fire properties according to the Balbi model
 */
#include "Fire.h"

struct Fire::FireProperties Fire ::balbi(FuelProperties *fuel, float u_mid, float v_mid, float x_norm, float y_norm, float x_slope, float y_slope, float fmc_g)
{
    // struct to hold computed fire properties
    struct FireProperties fp;
    // Fuel Properties
    float oneHour = fuel->oneHour;///< one hour fuel load [t/ac]
    float tenHour = fuel->tenHour;///< ten hour fuel load [t/ac]
    float hundredHour = fuel->hundredHour;///< hundred hour fuel load [t/ac]
    float liveHerb = fuel->liveHerb;///< live herb fuel load [t/ac]
    float liveWoody = fuel->liveWoody;///< live woody fuel load [t/ac]
    float savOneHour = fuel->savOneHour;///< surface area to volume ratio of one hour fuels
    float savHerb = fuel->savHerb;///< surface area to volume ratio of live herbacious fuel load
    float savWoody = fuel->savWoody;///< surface area to volume ratio of live woody fuel load
    float fueldepthm = fuel->fuelDepth * 0.3048;///< fuel bed depth [m]
    float cmbcnst = fuel->heatContent * 2326;///< heat content [J/kg]
    float savTenHour = 109;///< surface area to volume ratio of ten hour fuel load
    float savHundredHour = 30;///< surface area to volume ratio of hundred hour fuel load
    float rhoFuel = fuel->fuelDensity * 16.0185;///< ovendry fuel particle density [kg/m^3]
    float fgi = (oneHour + liveHerb + liveWoody) * 0.2471;///< Initial fine fuel load [kg/m^2]
    float savr = (savOneHour*oneHour + savHerb*liveHerb + savWoody*liveWoody)/(oneHour + liveHerb + liveWoody); ///< Characteristic fine fuel load surface area to volume ratio
    float savrT = (savOneHour * oneHour + savTenHour * tenHour + savHundredHour * hundredHour + savHerb * liveHerb + savWoody * liveWoody)
               / (oneHour + tenHour + hundredHour + liveHerb + liveWoody);///<Characteristic SAV of total fuel class for residence time

    if (fgi < 0.00001) {
        // set fire properties
        fp.w = 0;
        fp.h = 0;
        fp.d = 0;
        fp.r = 0;
        fp.T = 0;
        fp.tau = 5000;
        fp.K = 0;
        fp.H0 = 0;
    } else {
        // universal constants
        float g = 9.81;///< gravity
        float pi = 3.14159265358979323846;///< pi
        float s = 17;///< stoichiometric constant - Balbi 2018
        float Chi_0 = 0.3;///< thin flame radiant fraction - Balbi 2009
        float B = 5.67e-8;///< Stefan-Boltzman
        float Deltah_v = 2.257e6;///< water evap enthalpy [J/kg]
        float C_p = 2e3;///< calorific capacity [J/kg] - Balbi 2009
        float C_pa = 1150;///< specific heat of air [J/Kg/K]
        float tau_0 = 75591;///< residence time coefficient - Anderson 196?
        float tau = tau_0 / (savrT / 0.3048)+300;///MM-11-2-22
        // fuel constants
        float m = fmc_g;///< fuel particle moisture content [0-1]
	float transfer = liveHerb * (1.20 - cure) / .90;
        float sigma = (oneHour + transfer)* 0.2471;///< dead fine fuel load [kg/m^2]
        float sigmaT = fgi;///< total fine fuel load [kg/m^2]
        //float rhoFuel  = 1500;                    ///< fuel density [kg/m^3]
        float T_i = 600;///< ignition temp [k]

        // model parameters
        float beta = sigma / (fueldepthm * rhoFuel);///< packing ratio of dead fuel [eq.1]
        float betaT = sigmaT / (fueldepthm * rhoFuel);///< total packing ratio [eq.2]
        float SAV = savr / 0.3048;///< fine fuel surface area to volume ratio [m^2/m^3]
        float lai = (SAV * fueldepthm * beta) / 2;///< leaf Area Index for dead fuel [eq.3]
        float nu = fmin(2 * lai, 2 * pi * beta / betaT);///< absorption coefficient [eq.5]
        float lv = fueldepthm;///< fuel length [m] ?? need better parameterization here
        float K1 = 100;///< drag force coefficient: 100 for field, 1400 for lab
        float r_00 = 2.5e-5;///< model parameter

        // Environmental Constants
        float rhoAir = 1.125;///< air Density [Kg/m^3]
        float T_a = 289.15;///< air Temp [K]
        float alpha = sqrt(x_slope * x_slope + y_slope * y_slope);///< slope angle [rad]
        float psi = 0;///< angle between wind and flame front [rad]
        float phi = 0;///< angle between flame front vector and slope vector [rad]

        int s_c = 0;
        int n_c = 0;
        int v_c = 0;
        if (abs(x_slope) < 0.001 and abs(y_slope) < 0.001) {
            s_c = 1;
        }
        if (abs(x_norm) < 0.001 and abs(y_norm) < 0.001) {
            n_c = 1;
        }
        if (abs(u_mid) < 0.001 and abs(v_mid) < 0.001) {
            v_c = 1;
        }
        if (s_c == 1 or n_c == 1) {
            phi = 0;
        } else {
            phi = acos((x_slope * x_norm + y_slope * y_norm) / (sqrt(x_slope * x_slope + y_slope * y_slope) * sqrt(x_norm * x_norm + y_norm * y_norm)));
	    if (isnan(phi)){
	      //std::cout<<"ROS error, phi isnan"<<std::endl;
	      phi = 0;
	    }
        }
        if (v_c == 1 or n_c == 1) {
            psi = 0;
        } else {
            psi = acos((u_mid * x_norm + v_mid * y_norm) / (sqrt(u_mid * u_mid + v_mid * v_mid) * sqrt(x_norm * x_norm + y_norm * y_norm)));
	    if (isnan(psi)){
	      //std::cout<<"ROS error, psi isnan"<<std::endl;
	      psi = 0;
	    }
        
        }
        if (phi > 0.785) {
            alpha = -alpha;
        }
        float KDrag = K1 * betaT * fmin(fueldepthm / lv, 1);///< Drag force coefficient [eq.7]
        float q = C_p * (T_i - T_a) + m * Deltah_v;///< Activation energy [eq.14]
        float A = fmin(SAV / (2 * pi), beta / betaT) * Chi_0 * cmbcnst / (4 * q);///< Radiant coefficient [eq.13]
        // Initial guess = Rothermel ROS
        float R = rothermel(fuel, max(u_mid, v_mid), alpha, fmc_g);///< Total Rate of Spread (ROS) [m/s]
        // Initial tilt angle guess = slope angle
        float gamma = alpha;///< Flame tilt angle
        float maxIter = 100;
        float R_tol = 1e-5;
        float iter = 1;
        float error = 1;
        float R_old = R;
        // find spread rates
        float Chi;///< Radiative fraction [-]
        float TFlame;///< Flame Temp [K]
        float u0;///< Upward gas velocity [m/s]
        float H;///< Flame height [m]
        float b;///< Convective coefficient [-]
        float ROSBase;///< ROS from base radiation [m/s]
        float ROSFlame;///< ROS from flame radiation [m/s]
        float ROSConv;///< ROS from convection [m/s]
        float V_mid = sqrt(u_mid * u_mid + v_mid * v_mid);///< Midflame Wind Velocity [m/s]
        while (iter < maxIter && error > R_tol) {
            Chi = Chi_0 / (1 + R * cos(gamma) / (SAV * r_00));//[eq.20]
            TFlame = T_a + cmbcnst * (1 - Chi) / ((s + 1) * C_pa);//[eq.16]
            u0 = 2 * nu * ((s + 1) / tau_0) * (rhoFuel / rhoAir) * (TFlame / T_a);//[eq.19]
            gamma = atan(tan(alpha) * cos(phi) + V_mid * cos(psi) / u0);
            H = u0 * u0 / (g * (TFlame / T_a - 1) * cos(alpha) * cos(alpha));//[eq.17]
            b = 1 / (q * tau_0 * u0 * betaT) * Deltah_v * nu * fmin(s / 30, 1);//[eq.8]
            /** 
            * Compute ROS
            */
            // ROS from base radiation
            ROSBase = fmin(SAV * fueldepthm * betaT / pi, 1) * (beta / betaT) * (beta / betaT) * (B * TFlame * TFlame * TFlame * TFlame) / (beta * rhoFuel * q);
            // ROS from flame radiation [eq.11]
            ROSFlame = A * R * (1 + sin(gamma) - cos(gamma)) / (1 + R * cos(gamma) / (SAV * r_00));
            // ROS from convection
            ROSConv = b * (tan(alpha) + 2 * V_mid / u0 * exp(-KDrag * R)) * cos(psi);
            // Total ROS
            R = ROSBase + ROSFlame + ROSConv;
            if (R < ROSBase) {
            R = ROSBase;
        }
        error = std::abs(R - R_old);
        iter++;
        R_old = R;
    }
	if (isnan(R)){
	  std::cout<<"R isnan"<<std::endl;
	  std::cout<<"psi = "<<psi<<", phi = "<<phi<<", alpha = "<<alpha<<std::endl;
	  std::cout<<"gamma = "<<gamma<<std::endl;
	  std::cout<<"xNorm = "<<x_norm<<", yNorm = "<<y_norm<<std::endl;
	  std::cout<<"u = "<<u_mid<<", v = "<<v_mid<<std::endl;
	  
	}
    // calculate flame depth
    float L = R * tau;
    if (isnan(L)) {
        L = dx;
    }
    float faw = (dx+dy)/2; //fire front cell width (width of cell)
    float fal = L < dx ? L : dx; // fire front length
    float Q0 = cmbcnst * fgi * faw*fal / tau;
    float H0 = (1.0 - Chi) * Q0;
    // set fire properties
    fp.w = u0;
    fp.h = H;
    fp.d = L;
    fp.r = R;
    fp.T = TFlame;
    fp.tau = tau;
    fp.K = KDrag;
    fp.H0 = H0;
  }
  return fp;
}
