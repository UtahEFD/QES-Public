//
//  TURBData.cpp
//
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//  Modified by Fabien Margairaz

#include <iostream>
#include "TURBGeneralData.h"

using namespace netCDF;
using namespace netCDF::exceptions;


//TURBGeneralData::TURBGeneralData(Args* arguments, WINDSGeneralData* WGD){
TURBGeneralData::TURBGeneralData(Args *arguments, WINDSGeneralData *WGD)
{

  std::cout << "[TURB Data] \t Loading QES-turb fields " << std::endl;

  // fullname passed to WINDSGeneralData
  input = new NetCDFInput(arguments->inputTURBFile);

  // nx,ny - face centered value (consistant with QES-Winds)
  input->getDimensionSize("x", nx);
  input->getDimensionSize("y", ny);
  // nz - face centered value + bottom ghost (consistant with QES-Winds)
  input->getDimensionSize("z", nz);
  // nt - number of time instance in data
  input->getDimensionSize("t", nt);

  //get time variables
  t.resize(nt);
  input->getVariableData("t", t);

  // make local copy of grid information
  // nx,ny,nz consitant with QES-Winds (face-center)
  nx = WGD->nx;
  ny = WGD->ny;
  nz = WGD->nz;

  float dx = WGD->dx;
  float dy = WGD->dy;
  float dz = WGD->dz;

  // x-grid (face-center & cell-center)
  x_fc.resize(nx, 0);
  x_cc.resize(nx - 1, 0);

  // y-grid (face-center & cell-center)
  y_fc.resize(ny, 0);
  y_cc.resize(ny - 1, 0);

  // z-grid (face-center & cell-center)
  z_fc.resize(nz, 0);
  z_cc.resize(nz - 1, 0);

  // x cell-center
  x_cc = WGD->x;
  // x face-center (this assume constant dx for the moment, same as QES-Winds)
  for (int i = 1; i < nx - 1; i++) {
    x_fc[i] = 0.5 * (WGD->x[i - 1] + WGD->x[i]);
  }
  x_fc[0] = x_fc[1] - dx;
  x_fc[nx - 1] = x_fc[nx - 2] + dx;

  // y cell-center
  y_cc = WGD->y;
  // y face-center (this assume constant dy for the moment, same as QES-winds)
  for (int i = 1; i < ny - 1; i++) {
    y_fc[i] = 0.5 * (WGD->y[i - 1] + WGD->y[i]);
  }
  y_fc[0] = y_fc[1] - dy;
  y_fc[ny - 1] = y_fc[ny - 2] + dy;

  // z cell-center
  z_cc = WGD->z;
  // z face-center (with ghost cell under the ground)
  for (int i = 1; i < nz; i++) {
    z_fc[i] = WGD->z_face[i - 1];
  }
  z_fc[0] = z_fc[1] - dz;

  // unused: int np_fc = nz*ny*nx;
  int np_cc = (nz - 1) * (ny - 1) * (nx - 1);

  // comp of the stress tensor
  txx.resize(np_cc, 0);
  txy.resize(np_cc, 0);
  txz.resize(np_cc, 0);
  tyy.resize(np_cc, 0);
  tyz.resize(np_cc, 0);
  tzz.resize(np_cc, 0);

  // derived turbulence quantities
  tke.resize(np_cc, 0);
  CoEps.resize(np_cc, 0);
}

void TURBGeneralData::loadNetCDFData(int stepin)
{

  std::cout << "[TURB Data] \t loading data at step " << stepin << std::endl;

  // netCDF variables
  std::vector<size_t> start;
  std::vector<size_t> count_cc;
  std::vector<size_t> count_fc;

  start = { static_cast<unsigned long>(stepin), 0, 0, 0 };
  count_cc = { 1,
    static_cast<unsigned long>(nz - 1),
    static_cast<unsigned long>(ny - 1),
    static_cast<unsigned long>(nx - 1) };

  // stress tensor
  input->getVariableData("txx", start, count_cc, txx);
  input->getVariableData("txy", start, count_cc, txy);
  input->getVariableData("txz", start, count_cc, txz);
  input->getVariableData("tyy", start, count_cc, tyy);
  input->getVariableData("tyz", start, count_cc, tyz);
  input->getVariableData("tzz", start, count_cc, tzz);

  // face-center variables
  input->getVariableData("tke", start, count_cc, tke);
  input->getVariableData("CoEps", start, count_cc, CoEps);

  return;
}
