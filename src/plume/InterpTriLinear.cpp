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

InterpTriLinear::InterpTriLinear(qes::Domain domain_in, bool debug_val = 0)
  : Interp(domain_in)
{
  // std::cout << "[InterpTriLinear] \t Setting InterpTriLinear fields " << std::endl;

  // copy debug information
  debug = debug_val;

  if (debug) {
    std::cout << "[InterpTriLinear] \t DEBUG - Domain boundary" << std::endl;
    std::cout << "\t\t xStart=" << xStart << " xEnd=" << xEnd << std::endl;
    std::cout << "\t\t yStart=" << yStart << " yEnd=" << yEnd << std::endl;
    std::cout << "\t\t zStart=" << zStart << " zEnd=" << zEnd << std::endl;
  }
}

void InterpTriLinear::interpWindsValues(const WINDSGeneralData *WGD,
                                        const vec3 &pos,
                                        vec3 &vel_out)
{
  // these are the current interp3D variables, as they are used for multiple interpolations for each particle
  interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

  // set interpolation indexing variables for uFace variables
  setInterp3Dindex_uFace(pos, wgt);
  // interpolation of variables on uFace
  interp3D_faceVar(WGD->u, wgt, vel_out._1);

  // set interpolation indexing variables for vFace variables
  setInterp3Dindex_vFace(pos, wgt);
  // interpolation of variables on vFace
  interp3D_faceVar(WGD->v, wgt, vel_out._2);

  // set interpolation indexing variables for wFace variables
  setInterp3Dindex_wFace(pos, wgt);
  // interpolation of variables on wFace
  interp3D_faceVar(WGD->w, wgt, vel_out._3);
}

void InterpTriLinear::interpTurbValues(const TURBGeneralData *TGD,
                                       const vec3 &pos,
                                       mat3sym &tau_out,
                                       vec3 &flux_div_out,
                                       float &nuT_out,
                                       float &CoEps_out)
{
  // these are the current interp3D variables, as they are used for multiple interpolations for each particle
  interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };
  // this replaces the old indexing trick, set the indexing variables for the interp3D for each particle,
  // then get interpolated values from the InterpTriLinear grid to the particle Lagrangian values for multiple datatype
  setInterp3Dindex_cellVar(pos, wgt);

  // this is the CoEps for the particle
  interp3D_cellVar(TGD->CoEps, wgt, CoEps_out);
  // make sure CoEps is always bigger than zero
  if (CoEps_out <= 1e-6) {
    CoEps_out = 1e-6;
  }

  // this is the current reynolds stress tensor
  interp3D_cellVar(TGD->txx, wgt, tau_out._11);
  interp3D_cellVar(TGD->txy, wgt, tau_out._12);
  interp3D_cellVar(TGD->txz, wgt, tau_out._13);
  interp3D_cellVar(TGD->tyy, wgt, tau_out._22);
  interp3D_cellVar(TGD->tyz, wgt, tau_out._23);
  interp3D_cellVar(TGD->tzz, wgt, tau_out._33);

  interp3D_cellVar(TGD->div_tau_x, wgt, flux_div_out._1);
  interp3D_cellVar(TGD->div_tau_y, wgt, flux_div_out._2);
  interp3D_cellVar(TGD->div_tau_z, wgt, flux_div_out._3);

  interp3D_cellVar(TGD->nuT, wgt, nuT_out);
}

void InterpTriLinear::interpTurbInitialValues(const TURBGeneralData *TGD,
                                              const vec3 &pos,
                                              mat3sym &tau_out,
                                              vec3 &sig_out)
{
  // these are the current interp3D variables, as they are used for multiple interpolations for each particle
  interpWeight wgt{ 0, 0, 0, 0.0, 0.0, 0.0 };

  // this replaces the old indexing trick, set the indexing variables for the
  // interp3D for each particle, then get interpolated values from the
  // InterpTriLinear grid to the particle values for multiple datatype
  setInterp3Dindex_cellVar(pos, wgt);

  // get the tau values from the InterpTriLinear grid for the particle value
  interp3D_cellVar(TGD->txx, wgt, tau_out._11);
  interp3D_cellVar(TGD->txy, wgt, tau_out._12);
  interp3D_cellVar(TGD->txz, wgt, tau_out._13);
  interp3D_cellVar(TGD->tyy, wgt, tau_out._22);
  interp3D_cellVar(TGD->tyz, wgt, tau_out._23);
  interp3D_cellVar(TGD->tzz, wgt, tau_out._33);

  sig_out._1 = std::sqrt(std::abs(tau_out._11));
  if (sig_out._1 == 0.0)
    sig_out._1 = 1e-8;
  sig_out._2 = std::sqrt(std::abs(tau_out._22));
  if (sig_out._2 == 0.0)
    sig_out._2 = 1e-8;
  sig_out._3 = std::sqrt(std::abs(tau_out._33));
  if (sig_out._3 == 0.0)
    sig_out._3 = 1e-8;
}


void InterpTriLinear::setInterp3Dindex_uFace(const vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // float par_x = par_xPos - xStart + 1.0*dx;
  // float par_y = par_yPos - yStart + 0.5*dy;
  // float par_z = par_zPos - zStart + 0.5*dz;
  float par_x = pos._1 - 0.0f * dx;
  float par_y = pos._2 - 0.5f * dy;
  // float par_z = par_zPos + 0.5 * dz;

  wgt.ii = std::floorf(par_x / (dx + 1e-7));
  wgt.jj = std::floorf(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - std::floorf(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - std::floorf(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  auto itr = std::lower_bound(domain.z.begin(), domain.z.end(), pos._3);
  wgt.kk = itr - domain.z.begin() - 1;

  wgt.kw = (pos._3 - domain.z[wgt.kk]) / (domain.z[wgt.kk + 1] - domain.z[wgt.kk]);
}


void InterpTriLinear::setInterp3Dindex_vFace(const vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // float par_x = par_xPos - xStart + 0.5*dx;
  // float par_y = par_yPos - yStart + 1.0*dy;
  // float par_z = par_zPos - zStart + 0.5*dz;
  float par_x = pos._1 - 0.5f * dx;
  float par_y = pos._2 - 0.0f * dy;
  // float par_z = par_zPos + 0.5 * dz;

  wgt.ii = std::floorf(par_x / (dx + 1e-7));
  wgt.jj = std::floorf(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - std::floorf(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - std::floorf(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-4));

  auto itr = std::lower_bound(domain.z.begin(), domain.z.end(), pos._3);
  wgt.kk = itr - domain.z.begin() - 1;

  wgt.kw = (pos._3 - domain.z[wgt.kk]) / (domain.z[wgt.kk + 1] - domain.z[wgt.kk]);
}

void InterpTriLinear::setInterp3Dindex_wFace(const vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // float par_x = par_xPos - xStart + 0.5*dx;
  // float par_y = par_yPos - yStart + 0.5*dy;
  // float par_z = par_zPos - zStart + 1.0*dz;
  float par_x = pos._1 - 0.5f * dx;
  float par_y = pos._2 - 0.5f * dy;
  // float par_z = par_zPos + 1.0 * dz;

  wgt.ii = floor(par_x / (dx + 1e-7));
  wgt.jj = floor(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - std::floorf(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - std::floorf(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  auto itr = std::lower_bound(domain.z_face.begin(), domain.z_face.end(), pos._3);
  wgt.kk = itr - domain.z_face.begin() - 1;

  wgt.kw = (pos._3 - domain.z_face[wgt.kk]) / (domain.z_face[wgt.kk + 1] - domain.z_face[wgt.kk]);
}

// always call this after setting the interpolation indices with the setInterp3Dindex_u/v/wFace() function!
void InterpTriLinear::interp3D_faceVar(const std::vector<float> &EulerData,
                                       const interpWeight &wgt,
                                       float &out)
{

  float cube[2][2][2] = { 0.0 };

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
  float u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                + wgt.iw * wgt.jw * cube[1][1][0]
                + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  float u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
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
        long idx = (wgt.kk + kkk) * (ny * nx) + (wgt.jj + jjj) * (nx) + (wgt.ii + iii);
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
void InterpTriLinear::setInterp3Dindex_cellVar(const vec3 &pos, interpWeight &wgt)
{

  // set a particle position corrected by the start of the domain in each direction
  // the algorithm assumes the list starts at x = 0.

  float par_x = pos._1 - 0.5f * dx;
  float par_y = pos._2 - 0.5f * dy;
  // double par_z = par_zPos + 0.5 * dz;

  // double par_x = par_xPos - xStart + 0.5*dx;
  // double par_y = par_yPos - yStart + 0.5*dy;
  // double par_z = par_zPos - zStart + 0.5*dz;

  // index of the nearest node in negative direction
  wgt.ii = std::floorf(par_x / (dx + 1e-7));
  wgt.jj = std::floorf(par_y / (dy + 1e-7));
  // wgt.kk = floor(par_z / (dz + 1e-7));

  // fractional distance between the nearest nodes
  wgt.iw = (par_x / dx) - std::floorf(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - std::floorf(par_y / (dy + 1e-7));
  // wgt.kw = (par_z / dz) - floor(par_z / (dz + 1e-7));

  auto itr = std::lower_bound(domain.z.begin(), domain.z.end(), pos._3);
  wgt.kk = itr - domain.z.begin() - 1;

  wgt.kw = (pos._3 - domain.z[wgt.kk]) / (domain.z[wgt.kk + 1] - domain.z[wgt.kk]);
}


// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
void InterpTriLinear::interp3D_cellVar(const std::vector<float> &EulerData,
                                       const interpWeight &wgt,
                                       float &out)
{


  // now set the cube to zero, then fill it using the indices and the counters from the indices
  float cube[2][2][2] = { 0.0 };

  // now set the cube values
  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        // set the actual indices to use for the linearized Euler data
        long idx = (wgt.kk + kkk) * (ny - 1) * (nx - 1) + (wgt.jj + jjj) * (nx - 1) + (wgt.ii + iii);
        cube[iii][jjj][kkk] = EulerData.at(idx);
      }
    }
  }

  // now do the interpolation, with the cube, the counters from the indices,
  // and the normalized width between the point locations and the closest cell left walls
  float u_low = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
                + wgt.iw * (1 - wgt.jw) * cube[1][0][0]
                + wgt.iw * wgt.jw * cube[1][1][0]
                + (1 - wgt.iw) * wgt.jw * cube[0][1][0];
  float u_high = (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
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
        long idx = (wgt.kk + kkk) * (ny - 1) * (nx - 1) + (wgt.jj + jjj) * (nx - 1) + (wgt.ii + iii);
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
