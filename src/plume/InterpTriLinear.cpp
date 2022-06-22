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

/** @file InterpTriLinear.cpp */

#include "InterpTriLinear.h"


InterpTriLinear::InterpTriLinear(WINDSGeneralData *WGD, TURBGeneralData *TGD, const bool &debug_val)
  : Interp(WGD)
{
  //std::cout << "[InterpTriLinear] \t Setting InterpTriLinear fields " << std::endl;

  // copy debug information
  debug = debug_val;

  if (debug == true) {
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
  // this replaces the old indexing trick, set the indexing variables for the
  // interp3D for each particle, then get interpolated values from the
  // InterpTriLinear grid to the particle values for multiple datatypes
  setInterp3Dindex_cellVar(xPos, yPos, zPos);

  // get the tau values from the InterpTriLinear grid for the particle value
  txx_out = interp3D_cellVar(TGD->txx);
  txy_out = interp3D_cellVar(TGD->txy);
  txz_out = interp3D_cellVar(TGD->txz);
  tyy_out = interp3D_cellVar(TGD->tyy);
  tyz_out = interp3D_cellVar(TGD->tyz);
  tzz_out = interp3D_cellVar(TGD->tzz);

  sig_x_out = std::sqrt(std::abs(txx_out));
  if (sig_x_out == 0.0)
    sig_x_out = 1e-8;
  sig_y_out = std::sqrt(std::abs(tyy_out));
  if (sig_y_out == 0.0)
    sig_y_out = 1e-8;
  sig_z_out = std::sqrt(std::abs(tzz_out));
  if (sig_z_out == 0.0)
    sig_z_out = 1e-8;

  return;
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
                                   double &CoEps_out)
{
  // set in`teroplation indexing variables for uFace variables
  setInterp3Dindex_uFace(xPos, yPos, zPos);
  // interpolation of variables on uFace
  uMean_out = interp3D_faceVar(WGD->u);
  //flux_div_x_out = interp3D_faceVar(dtxxdx);
  //flux_div_y_out = interp3D_faceVar(dtxydx);
  //flux_div_z_out = interp3D_faceVar(dtxzdx);

  // set interpolation indexing variables for vFace variables
  setInterp3Dindex_vFace(xPos, yPos, zPos);
  // interpolation of variables on vFace
  vMean_out = interp3D_faceVar(WGD->v);
  //flux_div_x_out += interp3D_faceVar(dtxydy);
  //flux_div_y_out += interp3D_faceVar(dtyydy);
  //flux_div_z_out += interp3D_faceVar(dtyzdy);

  // set interpolation indexing variables for wFace variables
  setInterp3Dindex_wFace(xPos, yPos, zPos);
  // interpolation of variables on wFace
  wMean_out = interp3D_faceVar(WGD->w);
  //flux_div_x_out += interp3D_faceVar(dtxzdz);
  //flux_div_y_out += interp3D_faceVar(dtyzdz);
  //flux_div_z_out += interp3D_faceVar(dtzzdz);

  // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
  // then get interpolated values from the InterpTriLinear grid to the particle Lagrangian values for multiple datatypes
  setInterp3Dindex_cellVar(xPos, yPos, zPos);

  // this is the Co times Eps for the particle
  // LA note: because Bailey's code uses Eps by itself and this does not, I wanted an option to switch between the two if necessary
  //  it's looking more and more like we will just use CoEps.
  CoEps_out = interp3D_cellVar(TGD->CoEps);
  // make sure CoEps is always bigger than zero
  if (CoEps_out <= 1e-6) {
    CoEps_out = 1e-6;
  }

  // this is the current reynolds stress tensor
  txx_out = interp3D_cellVar(TGD->txx);
  txy_out = interp3D_cellVar(TGD->txy);
  txz_out = interp3D_cellVar(TGD->txz);
  tyy_out = interp3D_cellVar(TGD->tyy);
  tyz_out = interp3D_cellVar(TGD->tyz);
  tzz_out = interp3D_cellVar(TGD->tzz);


  flux_div_x_out = interp3D_cellVar(TGD->div_tau_x);
  flux_div_y_out = interp3D_cellVar(TGD->div_tau_y);
  flux_div_z_out = interp3D_cellVar(TGD->div_tau_z);
  return;
}

void InterpTriLinear::setInterp3Dindex_uFace(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // set a particle position corrected by the start of the domain in each direction
  //double par_x = par_xPos - xStart + 1.0*dx;
  //double par_y = par_yPos - yStart + 0.5*dy;
  //double par_z = par_zPos - zStart + 0.5*dz;
  double par_x = par_xPos - 0.0 * dx;
  double par_y = par_yPos - 0.5 * dy;
  double par_z = par_zPos + 0.5 * dz;

  ii = floor(par_x / (dx + 1e-7));
  jj = floor(par_y / (dy + 1e-7));
  kk = floor(par_z / (dz + 1e-7));

  // fractional distance between nearest nodes
  iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  jw = (par_y / dy) - floor(par_y / (dy + 1e-7));
  kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  return;
}


void InterpTriLinear::setInterp3Dindex_vFace(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // set a particle position corrected by the start of the domain in each direction
  //double par_x = par_xPos - xStart + 0.5*dx;
  //double par_y = par_yPos - yStart + 1.0*dy;
  //double par_z = par_zPos - zStart + 0.5*dz;
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.0 * dy;
  double par_z = par_zPos + 0.5 * dz;

  ii = floor(par_x / (dx + 1e-7));
  jj = floor(par_y / (dy + 1e-7));
  kk = floor(par_z / (dz + 1e-7));

  // fractional distance between nearest nodes
  iw = (par_x / dx) - floor(par_x / (dx + 1e-4));
  jw = (par_y / dy) - floor(par_y / (dy + 1e-4));
  kw = (par_z / dz) - floor(par_z / (dz + 1e-4));

  return;
}

void InterpTriLinear::setInterp3Dindex_wFace(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // set a particle position corrected by the start of the domain in each direction
  //double par_x = par_xPos - xStart + 0.5*dx;
  //double par_y = par_yPos - yStart + 0.5*dy;
  //double par_z = par_zPos - zStart + 1.0*dz;
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;
  double par_z = par_zPos + 1.0 * dz;

  ii = floor(par_x / (dx + 1e-7));
  jj = floor(par_y / (dy + 1e-7));
  kk = floor(par_z / (dz + 1e-7));

  // fractional distance between nearest nodes
  iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  jw = (par_y / dy) - floor(par_y / (dy + 1e-7));
  kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  return;
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
double InterpTriLinear::interp3D_faceVar(const std::vector<float> &EulerData)
{

  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (kk + kkk) * (ny * nx) + (jj + jjj) * (nx) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
double InterpTriLinear::interp3D_faceVar(const std::vector<double> &EulerData)
{

  double cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        int idx = (kk + kkk) * (ny * nx) + (jj + jjj) * (nx) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}

// this gets around the problem of repeated or not repeated information, just needs called once before each interpolation,
// then intepolation on all kinds of datatypes can be done
void InterpTriLinear::setInterp3Dindex_cellVar(const double &par_xPos, const double &par_yPos, const double &par_zPos)
{

  // the next steps are to figure out the right indices to grab the values for cube from the data,
  // where indices are forced to be special if nx, ny, or nz are zero.
  // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

  // so this is called once before calling the interp3D function on many different datatypes
  // sets the current indices for grabbing the cube values and for interpolating with the cube,
  // but importantly sets ip,jp, and kp to zero if the number of cells in a dimension is 1
  // as this avoids referencing outside of array problems in an efficient manner
  // it also causes some stuff to be multiplied by zero so that interpolation works on any size of data without lots of if statements


  // set a particle position corrected by the start of the domain in each direction
  // the algorythm assumes the list starts at x = 0.

  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;
  double par_z = par_zPos + 0.5 * dz;

  //double par_x = par_xPos - xStart + 0.5*dx;
  //double par_y = par_yPos - yStart + 0.5*dy;
  //double par_z = par_zPos - zStart + 0.5*dz;

  // index of nearest node in negative direction
  // by adding a really small number to dx, it stops it from putting
  // the stuff on the right wall of the cell into the next cell, and
  // puts everything from the left wall to the right wall of a cell
  // into the left cell. Makes it simpler for interpolation, as without this,
  // the interpolation would reference outside the array if the input position was exactly on
  // nx, ny, or nz.
  // basically adding a small number to dx shifts the indices so that instead of going
  // from 0 to nx - 1, they go from 0 to nx - 2. This means that ii + ip can at most be nx - 1
  // and only if a particle lands directly on the far boundary condition edge
  ii = floor(par_x / (dx + 1e-7));
  jj = floor(par_y / (dy + 1e-7));
  kk = floor(par_z / (dz + 1e-7));

  // fractional distance between nearest nodes
  iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  jw = (par_y / dy) - floor(par_y / (dy + 1e-7));
  kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  /* FM -> OBSOLETE (this is insure at the boundary condition level)
    // now check to make sure that the indices are within the InterpTriLinear grid domain
    // Notice that this no longer includes throwing an error if particles touch the far walls
    // because adding a small number to dx in the index calculation forces the index to be completely left side biased
    
    if( ii < 0 || ii+ip > nx-1 ) {
    std::cerr << "ERROR (InterpTriLinear::setInterp3Dindexing): particle x position is out of range! x = \"" << par_xPos 
         << "\" ii+ip = \"" << ii << "\"+\"" << ip << "\",   nx-1 = \"" << nx-1 << "\"" << std::endl;
        exit(1);
    }
    if( jj < 0 || jj+jp > ny-1 ) {
        std::cerr << "ERROR (InterpTriLinear::setInterp3Dindexing): particle y position is out of range! y = \"" << par_yPos 
            << "\" jj+jp = \"" << jj << "\"+\"" << jp << "\",   ny-1 = \"" << ny-1 << "\"" << std::endl;
        exit(1);
    }
    if( kk < 0 || kk+kp > nz-1 ) {
        std::cerr << "ERROR (InterpTriLinear::setInterp3Dindexing): particle z position is out of range! z = \"" << par_zPos 
            << "\" kk+kp = \"" << kk << "\"+\"" << kp << "\",   nz-1 = \"" << nz-1 << "\"" << std::endl;
        exit(1);
    }
    */
}


// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double InterpTriLinear::interp3D_cellVar(const std::vector<float> &EulerData)
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
        int idx = (kk + kkk) * (ny - 1) * (nx - 1) + (jj + jjj) * (nx - 1) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double InterpTriLinear::interp3D_cellVar(const std::vector<double> &EulerData)
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
        int idx = (kk + kkk) * (ny - 1) * (nx - 1) + (jj + jjj) * (nx - 1) + (ii + iii);
        cube[iii][jjj][kkk] = EulerData[idx];
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  double u_low = (1 - iw) * (1 - jw) * cube[0][0][0] + iw * (1 - jw) * cube[1][0][0] + iw * jw * cube[1][1][0] + (1 - iw) * jw * cube[0][1][0];
  double u_high = (1 - iw) * (1 - jw) * cube[0][0][1] + iw * (1 - jw) * cube[1][0][1] + iw * jw * cube[1][1][1] + (1 - iw) * jw * cube[0][1][1];

  return (u_high - u_low) * kw + u_low;
}
