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

/** @file Eulerian.h 
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

#include "GriddedDataType.h"
#include "PlumeInputData.hpp"

#include "src/winds/WINDSGeneralData.h"
#include "src/winds/TURBGeneralData.h"


class Eulerian : public GriddedDataType
{

public:
  // constructor
  // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the Eulerian grid values,
  // then calculates the tau gradients which are then used to calculate the flux_div grid values.
  Eulerian(PlumeInputData *, WINDSGeneralData *, TURBGeneralData *, const bool &);

  // other input variable
  //double C_0;// a copy of the TGD grid information. This is used to separate out CoEps into its separate parts when doing debug output

  //double vel_threshold;

  // these are the gradients of many components of tau in many different direction. Tau is still kept inside TGD
  // this is only the derivatives that matter for calculating the flux_div
  // since this is just a temporary variable for calculating flux_div, it may be better to move this into the functions directly at some point
  // units are probably m/s^2
  // notice that tau is a symmetric tensor, so the flux div for each direction is taking a given x, y, or z face,
  //  then calculating the derivative of each stress on that face in the given direction of that stress.
  //  because of symmetry, the storage names don't always show this, so the symmetrical names are given in comments to make it clearer.
  std::vector<double> dtxxdx;// dtxxdx
  std::vector<double> dtxydy;// dtxydy
  std::vector<double> dtxzdz;// dtxzdz

  std::vector<double> dtxydx;// dtyxdx
  std::vector<double> dtyydy;// dtyydy
  std::vector<double> dtyzdz;// dtyzdz

  std::vector<double> dtxzdx;// dtzxdx
  std::vector<double> dtyzdy;// dtzydy
  std::vector<double> dtzzdz;// dtzzyz

  // temporary storage of sigma_x,_y,_z
  std::vector<double> sig_x;
  std::vector<double> sig_y;
  std::vector<double> sig_z;

  void setData(WINDSGeneralData *, TURBGeneralData *);
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
  // these are the current interp3D variables, as they are used for multiple interpolations for each particle
  int ii;// this is the nearest cell index to the left in the x direction
  int jj;// this is the nearest cell index to the left in the y direction
  int kk;// this is the nearest cell index to the left in the z direction
  double iw;// this is the normalized distance to the nearest cell index to the left in the x direction
  double jw;// this is the normalized distance to the nearest cell index to the left in the y direction
  double kw;// this is the normalized distance to the nearest cell index to the left in the z direction

  void setInterp3Dindex_uFace(const double &, const double &, const double &);
  void setInterp3Dindex_vFace(const double &, const double &, const double &);
  void setInterp3Dindex_wFace(const double &, const double &, const double &);
  double interp3D_faceVar(const std::vector<float> &);
  double interp3D_faceVar(const std::vector<double> &);

  void setInterp3Dindex_cellVar(const double &, const double &, const double &);
  double interp3D_cellVar(const std::vector<float> &);
  double interp3D_cellVar(const std::vector<double> &);

  // these are for calculating the gradients more efficiently
  // LA future work: I keep wondering, since we never use the gradients again since they are just used to calculate flux_div,
  //  which is what is used instead at some point in time should we get rid of storage of the gradient datasets?
  //  I guess they are useful fçor debugging. Also, it would be tough to do the calculation as efficiently
  //  because each component would need passed each and every time.
  void setDX_1D(const TURBGeneralData *TGD, const int idx);
  void setDY_1D(const TURBGeneralData *TGD, const int idx);
  void setDZ_1D(const TURBGeneralData *TGD, const int idx);

  void setDX_Forward(const TURBGeneralData *TGD, const int idx);// second order forward differencing for calc gradient in the x direction of tau
  void setDY_Forward(const TURBGeneralData *TGD, const int idx);// second order forward differencing for calc gradient in the y direction of tau
  void setDZ_Forward(const TURBGeneralData *TGD, const int idx);// second order forward differencing for calc gradient in the z direction of tau

  void setDX_Backward(const TURBGeneralData *TGD, const int idx);// second order backward differencing for calc gradient in the x direction of tau
  void setDY_Backward(const TURBGeneralData *TGD, const int idx);// second order backward differencing for calc gradient in the y direction of tau
  void setDZ_Backward(const TURBGeneralData *TGD, const int idx);// second order backward differencing for calc gradient in the z direction of tau

  void setStressGradient(TURBGeneralData *);
  void setStressGrads(TURBGeneralData *);
  void setSigmas(TURBGeneralData *);
  double getMaxVariance(const std::vector<double> &, const std::vector<double> &, const std::vector<double> &);

  void setBC(WINDSGeneralData *, TURBGeneralData *);

  // timer class useful for debugging and timing different operations
  calcTime timers;

  // copies of debug related information from the input arguments
  bool debug;
};

inline void Eulerian::setDX_1D(const TURBGeneralData *TGD, const int idx)
{
  dtxxdx.at(idx) = 0.0;
  dtxydx.at(idx) = 0.0;
  dtxzdx.at(idx) = 0.0;
}

// second order forward differencing for calc gradient in the x direction of tau
inline void Eulerian::setDX_Forward(const TURBGeneralData *TGD, const int idx)
{
  int idx_xp1 = idx + 1;
  int idx_xp2 = idx + 2;

  dtxxdx.at(idx) = (-3.0 * TGD->txx.at(idx) + 4.0 * TGD->txx.at(idx_xp1) - TGD->txx.at(idx_xp2)) * 0.5 / dx;
  dtxydx.at(idx) = (-3.0 * TGD->txy.at(idx) + 4.0 * TGD->txy.at(idx_xp1) - TGD->txy.at(idx_xp2)) * 0.5 / dx;
  dtxzdx.at(idx) = (-3.0 * TGD->txz.at(idx) + 4.0 * TGD->txz.at(idx_xp1) - TGD->txz.at(idx_xp2)) * 0.5 / dx;
}

// second order backward differencing for calc gradient in the x direction of tau
inline void Eulerian::setDX_Backward(const TURBGeneralData *TGD, const int idx)
{
  int idx_xm1 = idx - 1;
  int idx_xm2 = idx - 2;

  dtxxdx.at(idx) = (3.0 * TGD->txx.at(idx) - 4.0 * TGD->txx.at(idx_xm1) + TGD->txx.at(idx_xm2)) * 0.5 / dx;
  dtxydx.at(idx) = (3.0 * TGD->txy.at(idx) - 4.0 * TGD->txy.at(idx_xm1) + TGD->txy.at(idx_xm2)) * 0.5 / dx;
  dtxzdx.at(idx) = (3.0 * TGD->txz.at(idx) - 4.0 * TGD->txz.at(idx_xm1) + TGD->txz.at(idx_xm2)) * 0.5 / dx;
}


inline void Eulerian::setDY_1D(const TURBGeneralData *TGD, const int idx)
{
  dtxydy.at(idx) = 0.0;
  dtyydy.at(idx) = 0.0;
  dtyzdy.at(idx) = 0.0;
}

// second order forward differencing for calc gradient in the y direction of tau
inline void Eulerian::setDY_Forward(const TURBGeneralData *TGD, const int idx)
{
  int idx_yp1 = idx + (nx - 1);
  int idx_yp2 = idx + 2 * (nx - 1);

  dtxydy.at(idx) = (-3.0 * TGD->txy.at(idx) + 4.0 * TGD->txy.at(idx_yp1) - TGD->txy.at(idx_yp2)) * 0.5 / dy;
  dtyydy.at(idx) = (-3.0 * TGD->tyy.at(idx) + 4.0 * TGD->tyy.at(idx_yp1) - TGD->tyy.at(idx_yp2)) * 0.5 / dy;
  dtyzdy.at(idx) = (-3.0 * TGD->tyz.at(idx) + 4.0 * TGD->tyz.at(idx_yp1) - TGD->tyz.at(idx_yp2)) * 0.5 / dy;
}

// second order backward differencing for calc gradient in the y direction of tau
inline void Eulerian::setDY_Backward(const TURBGeneralData *TGD, const int idx)
{
  int idx_ym1 = idx - (nx - 1);
  int idx_ym2 = idx - 2 * (nx - 1);

  dtxydy.at(idx) = (3.0 * TGD->txy.at(idx) - 4.0 * TGD->txy.at(idx_ym1) + TGD->txy.at(idx_ym2)) * 0.5 / dy;
  dtyydy.at(idx) = (3.0 * TGD->tyy.at(idx) - 4.0 * TGD->tyy.at(idx_ym1) + TGD->tyy.at(idx_ym2)) * 0.5 / dy;
  dtyzdy.at(idx) = (3.0 * TGD->tyz.at(idx) - 4.0 * TGD->tyz.at(idx_ym1) + TGD->tyz.at(idx_ym2)) * 0.5 / dy;
}


inline void Eulerian::setDZ_1D(const TURBGeneralData *TGD, const int idx)
{
  dtxzdz.at(idx) = 0.0;
  dtyzdz.at(idx) = 0.0;
  dtzzdz.at(idx) = 0.0;
}

// second order forward differencing for calc gradient in the z direction of tau
inline void Eulerian::setDZ_Forward(const TURBGeneralData *TGD, const int idx)
{
  int idx_zp1 = idx + (ny - 1) * (nx - 1);
  int idx_zp2 = idx + 2 * (ny - 1) * (nx - 1);

  dtxzdz.at(idx) = (-3.0 * TGD->txz.at(idx) + 4.0 * TGD->txz.at(idx_zp1) - TGD->txz.at(idx_zp2)) * 0.5 / dz;
  dtyzdz.at(idx) = (-3.0 * TGD->tyz.at(idx) + 4.0 * TGD->tyz.at(idx_zp1) - TGD->tyz.at(idx_zp2)) * 0.5 / dz;
  dtzzdz.at(idx) = (-3.0 * TGD->tzz.at(idx) + 4.0 * TGD->tzz.at(idx_zp1) - TGD->tzz.at(idx_zp2)) * 0.5 / dz;
}

// second order backward differencing for calc gradient in the z direction of tau
inline void Eulerian::setDZ_Backward(const TURBGeneralData *TGD, const int idx)
{
  int idx_zm1 = idx - (ny - 1) * (nx - 1);
  int idx_zm2 = idx - 2 * (ny - 1) * (nx - 1);

  dtxzdz.at(idx) = (3.0 * TGD->txz.at(idx) - 4.0 * TGD->txz.at(idx_zm1) + TGD->txz.at(idx_zm2)) * 0.5 / dz;
  dtyzdz.at(idx) = (3.0 * TGD->tyz.at(idx) - 4.0 * TGD->tyz.at(idx_zm1) + TGD->tyz.at(idx_zm2)) * 0.5 / dz;
  dtzzdz.at(idx) = (3.0 * TGD->tzz.at(idx) - 4.0 * TGD->tzz.at(idx_zm1) + TGD->tzz.at(idx_zm2)) * 0.5 / dz;
}
