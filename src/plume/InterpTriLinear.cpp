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

/** @file InterpTriLinear.cpp */

#include "InterpTriLinear.h"

InterpTriLinear::InterpTriLinear(WINDSGeneralData *WGD, TURBGeneralData *TGD, const bool &debug_val)
  : Interp(WGD)
{
  // std::cout << "[InterpTriLinear] \t Setting InterpTriLinear fields " << std::endl;

  // copy debug information
  debug = debug_val;

  m_WGD = WGD;

  if (debug) {
    std::cout << "[InterpTriLinear] \t DEBUG - Domain boundary" << std::endl;
    std::cout << "\t\t xStart=" << xStart << " xEnd=" << xEnd << std::endl;
    std::cout << "\t\t yStart=" << yStart << " yEnd=" << yEnd << std::endl;
    std::cout << "\t\t zStart=" << zStart << " zEnd=" << zEnd << std::endl;
  }
}

void InterpTriLinear::interpInitialValues(const double &xPos,
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
  // these are the current interp3D variables, as they are used for multiple interpolations for each particle
  interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

  // this replaces the old indexing trick, set the indexing variables for the
  // interp3D for each particle, then get interpolated values from the
  // InterpTriLinear grid to the particle values for multiple datatype
  setInterp3Dindex_cellVar(xPos, yPos, zPos, wgt);

  // get the tau values from the InterpTriLinear grid for the particle value
  interp3D_cellVar(TGD->txx, wgt, txx_out);
  interp3D_cellVar(TGD->txy, wgt, txy_out);
  interp3D_cellVar(TGD->txz, wgt, txz_out);
  interp3D_cellVar(TGD->tyy, wgt, tyy_out);
  interp3D_cellVar(TGD->tyz, wgt, tyz_out);
  interp3D_cellVar(TGD->tzz, wgt, tzz_out);

  sig_x_out = std::sqrt(std::abs(txx_out));
  if (sig_x_out == 0.0)
    sig_x_out = 1e-8;
  sig_y_out = std::sqrt(std::abs(tyy_out));
  if (sig_y_out == 0.0)
    sig_y_out = 1e-8;
  sig_z_out = std::sqrt(std::abs(tzz_out));
  if (sig_z_out == 0.0)
    sig_z_out = 1e-8;
}

void InterpTriLinear::interpValues(const double &xPos,
                                   const double &yPos,
                                   const double &zPos,
                                   const WINDSGeneralData *WGD,
                                   double &uMean_out,
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
                                   double &nuT_out,
                                   double &CoEps_out)
{
  // these are the current interp3D variables, as they are used for multiple interpolations for each particle
  interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

  // set interpolation indexing variables for uFace variables
  setInterp3Dindex_uFace(xPos, yPos, zPos, wgt);
  // interpolation of variables on uFace
  interp3D_faceVar(WGD->u, wgt, uMean_out);

  // set interpolation indexing variables for vFace variables
  setInterp3Dindex_vFace(xPos, yPos, zPos, wgt);
  // interpolation of variables on vFace
  interp3D_faceVar(WGD->v, wgt, vMean_out);

  // set interpolation indexing variables for wFace variables
  setInterp3Dindex_wFace(xPos, yPos, zPos, wgt);
  // interpolation of variables on wFace
  interp3D_faceVar(WGD->w, wgt, wMean_out);

  // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
  // then get interpolated values from the InterpTriLinear grid to the particle Lagrangian values for multiple datatype
  setInterp3Dindex_cellVar(xPos, yPos, zPos, wgt);

  // this is the CoEps for the particle
  interp3D_cellVar(TGD->CoEps, wgt, CoEps_out);
  // make sure CoEps is always bigger than zero
  if (CoEps_out <= 1e-6) {
    CoEps_out = 1e-6;
  }

  // this is the current reynolds stress tensor
  interp3D_cellVar(TGD->txx, wgt, txx_out);
  interp3D_cellVar(TGD->txy, wgt, txy_out);
  interp3D_cellVar(TGD->txz, wgt, txz_out);
  interp3D_cellVar(TGD->tyy, wgt, tyy_out);
  interp3D_cellVar(TGD->tyz, wgt, tyz_out);
  interp3D_cellVar(TGD->tzz, wgt, tzz_out);


  interp3D_cellVar(TGD->div_tau_x, wgt, flux_div_x_out);
  interp3D_cellVar(TGD->div_tau_y, wgt, flux_div_y_out);
  interp3D_cellVar(TGD->div_tau_z, wgt, flux_div_z_out);

  interp3D_cellVar(TGD->nuT, wgt, nuT_out);
}

void InterpTriLinear::setInterp3Dindex_uFace(const double &par_xPos,
                                             const double &par_yPos,
                                             const double &par_zPos,
                                             interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // double par_x = par_xPos - xStart + 1.0*dx;
  // double par_y = par_yPos - yStart + 0.5*dy;
  // double par_z = par_zPos - zStart + 0.5*dz;
  double par_x = par_xPos - 0.0 * dx;
  double par_y = par_yPos - 0.5 * dy;
  // double par_z = par_zPos + 0.5 * dz;

  wgt.ii = floor(par_x / (dx + 1e-7));
  wgt.jj = floor(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  auto itr = std::lower_bound(m_WGD->z.begin(), m_WGD->z.end(), par_zPos);
  wgt.kk = itr - m_WGD->z.begin() - 1;

  wgt.kw = (par_zPos - m_WGD->z[wgt.kk]) / (m_WGD->z[wgt.kk + 1] - m_WGD->z[wgt.kk]);
}


void InterpTriLinear::setInterp3Dindex_vFace(const double &par_xPos,
                                             const double &par_yPos,
                                             const double &par_zPos,
                                             interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // double par_x = par_xPos - xStart + 0.5*dx;
  // double par_y = par_yPos - yStart + 1.0*dy;
  // double par_z = par_zPos - zStart + 0.5*dz;
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.0 * dy;
  // double par_z = par_zPos + 0.5 * dz;

  wgt.ii = floor(par_x / (dx + 1e-7));
  wgt.jj = floor(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-4));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-4));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-4));

  auto itr = std::lower_bound(m_WGD->z.begin(), m_WGD->z.end(), par_zPos);
  wgt.kk = itr - m_WGD->z.begin() - 1;

  wgt.kw = (par_zPos - m_WGD->z[wgt.kk]) / (m_WGD->z[wgt.kk + 1] - m_WGD->z[wgt.kk]);
}

void InterpTriLinear::setInterp3Dindex_wFace(const double &par_xPos,
                                             const double &par_yPos,
                                             const double &par_zPos,
                                             interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // double par_x = par_xPos - xStart + 0.5*dx;
  // double par_y = par_yPos - yStart + 0.5*dy;
  // double par_z = par_zPos - zStart + 1.0*dz;
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;
  // double par_z = par_zPos + 1.0 * dz;

  wgt.ii = floor(par_x / (dx + 1e-7));
  wgt.jj = floor(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  auto itr = std::lower_bound(m_WGD->z_face.begin(), m_WGD->z_face.end(), par_zPos);
  wgt.kk = itr - m_WGD->z_face.begin() - 1;

  wgt.kw = (par_zPos - m_WGD->z_face[wgt.kk]) / (m_WGD->z_face[wgt.kk + 1] - m_WGD->z_face[wgt.kk]);
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
void InterpTriLinear::interp3D_faceVar(const std::vector<float> &EulerData,
                                       const interpWeight &wgt,
                                       double &out)
{

  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (wgt.kk + kkk) * (ny * nx) + (wgt.jj + jjj) * (nx) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                 + wgt.iw * wgt.jw * cube[1][1][0]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  double u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                  + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                  + wgt.iw * wgt.jw * cube[1][1][1]
                  + (1 - wgt.iw) * wgt.jw * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
void InterpTriLinear::interp3D_faceVar(const std::vector<double> &EulerData,
                                       const interpWeight &wgt,
                                       double &out)
{

  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (wgt.kk + kkk) * (ny * nx) + (wgt.jj + jjj) * (nx) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                 + wgt.iw * wgt.jw * cube[1][1][0]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  double u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                  + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                  + wgt.iw * wgt.jw * cube[1][1][1]
                  + (1 - wgt.iw) * wgt.jw * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

// this gets around the problem of repeated or not repeated information, just needs called once before each interpolation,
// then interpolation on all kinds of datatype can be done
void InterpTriLinear::setInterp3Dindex_cellVar(const double &par_xPos,
                                               const double &par_yPos,
                                               const double &par_zPos,
                                               interpWeight &wgt)
{

  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // so this is called once before calling the interp3D function on many datatype
  // sets the current indices for grabbing the cube values and for interpolating with the cube,
  // but importantly sets ip,jp, and kp to zero if the number of cells in a dimension is 1
  // as this avoids referencing outside of array problems in an efficient manner
  // it also causes some stuff to be multiplied by zero so that interpolation works on any size of data without lots of if statements

  // set a particle position corrected by the start of the domain in each direction
  // the algorithm assumes the list starts at x = 0.

  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;
  // double par_z = par_zPos + 0.5 * dz;

  // double par_x = par_xPos - xStart + 0.5*dx;
  // double par_y = par_yPos - yStart + 0.5*dy;
  // double par_z = par_zPos - zStart + 0.5*dz;

  // index of the nearest node in negative direction
  // by adding a tiny number to dx, it stops it from putting
  // the stuff on the right wall of the cell into the next cell, and
  // puts everything from the left wall to the right wall of a cell
  // into the left cell. Makes it simpler for interpolation, as without this,
  // the interpolation would reference outside the array if the input position was exactly on
  // nx, ny, or nz.
  // basically adding a small number to dx shifts the indices so that instead of going
  // from 0 to nx - 1, they go from 0 to nx - 2. This means that ii + ip can at most be nx - 1
  // and only if a particle lands directly on the far boundary condition edge
  wgt.ii = floor(par_x / (dx + 1e-7));
  wgt.jj = floor(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  auto itr = std::lower_bound(m_WGD->z.begin(), m_WGD->z.end(), par_zPos);
  wgt.kk = itr - m_WGD->z.begin() - 1;

  wgt.kw = (par_zPos - m_WGD->z[wgt.kk]) / (m_WGD->z[wgt.kk + 1] - m_WGD->z[wgt.kk]);
}


// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
void InterpTriLinear::interp3D_cellVar(const std::vector<float> &EulerData,
                                       const interpWeight &wgt,
                                       double &out)
{

  // first set a cube of size two to zero.
  // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // now set the cube to zero, then fill it using the indices and the counters from the indices
  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (wgt.kk + kkk) * (ny - 1) * (nx - 1) + (wgt.jj + jjj) * (nx - 1) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                 + wgt.iw * wgt.jw * cube[1][1][0]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  double u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                  + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                  + wgt.iw * wgt.jw * cube[1][1][1]
                  + (1 - wgt.iw) * wgt.jw * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
void InterpTriLinear::interp3D_cellVar(const std::vector<double> &EulerData,
                                       const interpWeight &wgt,
                                       double &out)
{

  // first set a cube of size two to zero.
  // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // now set the cube to zero, then fill it using the indices and the counters from the indices
  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (wgt.kk + kkk) * (ny - 1) * (nx - 1) + (wgt.jj + jjj) * (nx - 1) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                 + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                 + wgt.iw * wgt.jw * cube[1][1][0]
                 + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  double u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
                  + wgt.iw * (1 - wgt.jw) * cube[1][0][1]
                  + wgt.iw * wgt.jw * cube[1][1][1]
                  + (1 - wgt.iw) * wgt.jw * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}
