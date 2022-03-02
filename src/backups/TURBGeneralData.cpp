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
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file TURBGeneralData.cpp
 * @brief :document this:
 */

#include "TURBGeneralData.h"

// TURBGeneralData::TURBGeneralData(Args* arguments, URBGeneralData* WGD){
TURBGeneralData::TURBGeneralData(const WINDSInputData *WID, WINDSGeneralData *WGDin)
{

  auto StartTime = std::chrono::high_resolution_clock::now();

  std::cout << "[QES-TURB]\t Initialization of turbulence model...\n";
  m_WGD = WGDin;

  // local copies of trubulence parameters
  turbUpperBound = WID->turbParams->turbUpperBound;

  if (WID->turbParams->sigConst) {
    Vector3 sigConst;
    sigConst = *(WID->turbParams->sigConst);

    sigUOrg = sigConst[0];
    sigVOrg = sigConst[1];
    sigWOrg = sigConst[2];

    sigUConst = 1.5 * sigUOrg * sigUOrg * cPope * cPope;
    sigVConst = 1.5 * sigVOrg * sigVOrg * cPope * cPope;
    sigWConst = 1.5 * sigWOrg * sigWOrg * cPope * cPope;
  }

  flagNonLocalMixing = WID->turbParams->flagNonLocalMixing;
  if (flagNonLocalMixing) {
    std::cout << "\t\t Non-Local mixing for buidlings: ON \n";
  }

  if (WID->simParams->verticalStretching > 0) {
    flagUniformZGrid = false;
  }

  backgroundMixing = WID->turbParams->backgroundMixing;

  // make local copy of grid information
  // nx,ny,nz consitant with WINDS (face-center)
  // WINDS->grid correspond to face-center grid
  nz = m_WGD->nz;
  ny = m_WGD->ny;
  nx = m_WGD->nx;

  dz = m_WGD->dz;
  dy = m_WGD->dy;
  dx = m_WGD->dx;

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
  x_cc = m_WGD->x;
  // x face-center (this assume constant dx for the moment, same as QES-winds)
  for (int i = 1; i < nx - 1; i++) {
    x_fc[i] = 0.5 * (m_WGD->x[i - 1] + m_WGD->x[i]);
  }
  x_fc[0] = x_fc[1] - dx;
  x_fc[nx - 1] = x_fc[nx - 2] + dx;

  // y cell-center
  y_cc = m_WGD->y;
  // y face-center (this assume constant dy for the moment, same as QES-winds)
  for (int i = 1; i < ny - 1; i++) {
    y_fc[i] = 0.5 * (m_WGD->y[i - 1] + m_WGD->y[i]);
  }
  y_fc[0] = y_fc[1] - dy;
  y_fc[ny - 1] = y_fc[ny - 2] + dy;

  // z cell-center
  z_cc = m_WGD->z;
  // z face-center (with ghost cell under the ground)
  for (int i = 1; i < nz; i++) {
    z_fc[i] = m_WGD->z_face[i - 1];
  }
  z_fc[0] = z_fc[1] - dz;

  // unused: int np_fc = nz*ny*nx;
  int np_cc = (nz - 1) * (ny - 1) * (nx - 1);

  iturbflag.resize(np_cc, 0);

  /*
     vector containing cell id of fluid cell
     do not include 1 cell shell around the domain
     => i=1...nx-2 j=1...ny-2
     do not include 1 cell layer at the top of the domain
     => k=0...nz-2
  */
  for (int k = 0; k < nz - 2; k++) {
    for (int j = 1; j < ny - 2; j++) {
      for (int i = 1; i < nx - 2; i++) {
        int id = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (m_WGD->icellflag[id] != 0 && m_WGD->icellflag[id] != 2) {
          icellfluid.push_back(id);
          iturbflag.at(id) = 1;
        }
      }
    }
  }


  // definition of the solid wall for loglaw
  std::cout << "\t\t Defining Solid Walls...\n";
  wallVec.push_back(new TURBWallBuilding(WID, m_WGD, this));
  wallVec.push_back(new TURBWallTerrain(WID, m_WGD, this));
  // std::cout << "\t\t Walls Defined...\n";

  // mixing length

  std::cout << "\t\t Defining Local Mixing Length...\n";
  auto mlStartTime = std::chrono::high_resolution_clock::now();
  if (WID->turbParams->methodLocalMixing == 0) {
    std::cout << "\t\t Default Local Mixing Length...\n";
    localMixing = new LocalMixingDefault();
  } else if (WID->turbParams->methodLocalMixing == 1) {
    std::cout << "\t\t Computing Local Mixing Length using serial code...\n";
    localMixing = new LocalMixingSerial();
  } else if (WID->turbParams->methodLocalMixing == 2) {
    /*******Add raytrace code here********/
    std::cout << "\t\t Computing mixing length scales..." << std::endl;
    // WID->simParams->DTE_mesh->calculateMixingLength(nx, ny, nz, dx, dy, dz, WGD->icellflag, WGD->mixingLengths);
  } else if (WID->turbParams->methodLocalMixing == 3) {
    localMixing = new LocalMixingOptix();
  } else if (WID->turbParams->methodLocalMixing == 4) {
    std::cout << "\t\t Loading Local Mixing Length data form NetCDF...\n";
    localMixing = new LocalMixingNetCDF();
  } else {
    // this should not happen (checked in TURBParams)
  }

  localMixing->defineMixingLength(WID, m_WGD);

  Lm.resize(np_cc, 0.0);
  // make a copy as mixing length will be modifiy by non local
  // (need to be reset at each time instances)
  for (auto id = 0u; id < icellfluid.size(); id++) {
    int idcc = icellfluid[id];
    Lm[idcc] = vonKar * m_WGD->mixingLengths[idcc];
  }

  auto mlEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> mlElapsed = mlEndTime - mlStartTime;
  std::cout << "\t\t Local Mixing Defined: elapsed time: " << mlElapsed.count() << " s\n";

  // Caluclating Turbulence quantities (u*,z0,d0) based on morphology of domain.
  bldgH_mean = 0.0;
  bldgH_max = 0.0;
  terrainH_max = 0.0;
  zRef = 0.0;
  uRef = 0.0;
  uStar = 0.0;

  std::cout << "\t\t Calculating Morphometric parametrization of trubulence..." << std::endl;

  if (WID->simParams->DTE_heightField) {
    terrainH_max = *max_element(m_WGD->terrain.begin(), m_WGD->terrain.end());
  } else {
    terrainH_max = 0.0;
  }

  // std::cout << "\t\t max terrain height = "<< terrainH_max << std::endl;

  // calculate the mean building h
  if (m_WGD->allBuildingsV.size() > 0) {
    float heffmax = 0.0;
    for (size_t i = 0; i < m_WGD->allBuildingsV.size(); i++) {
      bldgH_mean += m_WGD->allBuildingsV[m_WGD->building_id[i]]->H;
      heffmax = m_WGD->allBuildingsV[m_WGD->building_id[i]]->H;// height_eff;
      if (heffmax > bldgH_max) {
        bldgH_max = heffmax;
      }
    }
    bldgH_mean = bldgH_mean / float(m_WGD->allBuildingsV.size());

    // std::cout << "\t\t\t mean bldg height = "<< bldgH_mean << " max bldg height = "<< bldgH_max << std::endl;

    // Morphometric parametrization based on Grimmond and Oke (1999) and Kaster-Klein and Rotach (2003)
    // roughness length z0 as 0.1 mean building height
    z0d = 0.1 * bldgH_mean;
    // displacement height d0 as 0.7 mean building height
    d0d = 0.7 * bldgH_mean;

    // reference height as 3.0 mean building
    zRef = 3.0 * bldgH_mean;

  } else {
    z0d = WID->metParams->sensors[0]->TS[0]->site_z0;
    d0d = 0.0;
    zRef = 100.0 * z0d;
  }

  std::cout << "\t\t Computing friction velocity..." << std::endl;
  frictionVelocity();

  std::cout << "\t\t Allocating memory...\n";
  // comp. of the strain rate tensor
  /*
    Sxx.resize(np_cc, 0);
    Sxy.resize(np_cc, 0);
    Sxz.resize(np_cc, 0);
    Syy.resize(np_cc, 0);
    Syz.resize(np_cc, 0);
    Szz.resize(np_cc, 0);
  */
  // comp. of the velocity gradient tensor
  Gxx.resize(np_cc, 0);
  Gxy.resize(np_cc, 0);
  Gxz.resize(np_cc, 0);
  Gyx.resize(np_cc, 0);
  Gyy.resize(np_cc, 0);
  Gyz.resize(np_cc, 0);
  Gzx.resize(np_cc, 0);
  Gzy.resize(np_cc, 0);
  Gzz.resize(np_cc, 0);

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
  // std::cout << "\t\t Memory allocation completed.\n";

  if (flagCompDivStress) {
    // comp of the divergence of the stress tensor
    tmp_dtoxdx.resize(np_cc, 0);
    tmp_dtoydy.resize(np_cc, 0);
    tmp_dtozdz.resize(np_cc, 0);

    // comp of the divergence of the stress tensor
    div_tau_x.resize(np_cc, 0);
    div_tau_y.resize(np_cc, 0);
    div_tau_z.resize(np_cc, 0);
  }

  auto EndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> Elapsed = EndTime - StartTime;

  std::cout << "[QES-TURB]\t Initialization of turbulence model completed.\n";
  std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;
}

TURBGeneralData::TURBGeneralData(const std::string inputFile, WINDSGeneralData *WGDin)
{

  std::cout << "[TURB Data] \t Loading QES-turb fields " << std::endl;

  m_WGD = WGDin;

  // fullname passed to WINDSGeneralData
  input = new NetCDFInput(inputFile);

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
  nx = m_WGD->nx;
  ny = m_WGD->ny;
  nz = m_WGD->nz;

  float dx = m_WGD->dx;
  float dy = m_WGD->dy;
  float dz = m_WGD->dz;

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
  x_cc = m_WGD->x;
  // x face-center (this assume constant dx for the moment, same as QES-Winds)
  for (int i = 1; i < nx - 1; i++) {
    x_fc[i] = 0.5 * (m_WGD->x[i - 1] + m_WGD->x[i]);
  }
  x_fc[0] = x_fc[1] - dx;
  x_fc[nx - 1] = x_fc[nx - 2] + dx;

  // y cell-center
  y_cc = m_WGD->y;
  // y face-center (this assume constant dy for the moment, same as QES-winds)
  for (int i = 1; i < ny - 1; i++) {
    y_fc[i] = 0.5 * (m_WGD->y[i - 1] + m_WGD->y[i]);
  }
  y_fc[0] = y_fc[1] - dy;
  y_fc[ny - 1] = y_fc[ny - 2] + dy;

  // z cell-center
  z_cc = m_WGD->z;
  // z face-center (with ghost cell under the ground)
  for (int i = 1; i < nz; i++) {
    z_fc[i] = m_WGD->z_face[i - 1];
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

  // comp of the divergence of the stress tensor
  tmp_dtoxdx.resize(np_cc, 0);
  tmp_dtoydy.resize(np_cc, 0);
  tmp_dtozdz.resize(np_cc, 0);

  // comp of the divergence of the stress tensor
  div_tau_x.resize(np_cc, 0);
  div_tau_y.resize(np_cc, 0);
  div_tau_z.resize(np_cc, 0);
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

  divergenceStress();

  return;
}

// compute turbulence fields
void TURBGeneralData::run()
{

  auto StartTime = std::chrono::high_resolution_clock::now();

  std::cout << "[QES-TURB] \t Running turbulence model..." << std::endl;

  std::cout << "\t\t Computing friction velocity..." << std::endl;
  frictionVelocity();

  std::cout << "\t\t Computing Derivatives (Strain Rate)..." << std::endl;
  derivativeVelocity();
  // std::cout<<"\t\t Derivatives computed."<<std::endl;

  std::cout << "\t\t Computing Stess Tensor..." << std::endl;
  stressTensor();
  // std::cout<<"\t\t Stress Tensor computed."<<std::endl;

  if (flagNonLocalMixing) {
    std::cout << "\t\t Applying non-local mixing..." << std::endl;

    for (size_t i = 0; i < m_WGD->allBuildingsV.size(); i++) {
      m_WGD->allBuildingsV[m_WGD->building_id[i]]->NonLocalMixing(m_WGD, this, m_WGD->building_id[i]);
    }
    // std::cout<<"\t\t Non-local mixing completed."<<std::endl;
  }

  std::cout << "\t\t Checking Upper Bound of Turbulence Fields..." << std::endl;
  boundTurbFields();

  if (backgroundMixing > 0.0) {
    addBackgroundMixing();
  }

  if (flagCompDivStress) {
    divergenceStress();
  }

  auto EndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> Elapsed = EndTime - StartTime;

  std::cout << "[QES-TURB] \t Turbulence model completed.\n";
  std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;
}

void TURBGeneralData::frictionVelocity()
{
  float nVal = 0.0, uSum = 0.0;
  for (int j = 0; j < ny - 1; j++) {
    for (int i = 0; i < nx - 1; i++) {
      // search the vector for the first element with value 42
      std::vector<float>::iterator itr = std::lower_bound(z_fc.begin(), z_fc.end(), zRef);
      int k;
      if (itr != z_fc.end()) {
        k = itr - z_fc.begin();
        // std::cout << "\t\t\t ref height = "<< zRef << " kRef = "<< kRef << std::endl;
      } else {
        std::cerr << "[ERROR] Turbulence model : reference height is outside the domain" << std::endl;
        std::cerr << "\t Reference height zRef = " << zRef << " m." << std::endl;
        exit(EXIT_FAILURE);
      }

      int cellID = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
      int faceID = i + j * (nx) + k * (nx) * (ny);
      if (m_WGD->icellflag[cellID] != 0 && m_WGD->icellflag[cellID] != 2) {
        uSum += sqrt(pow(0.5 * (m_WGD->u[faceID] + m_WGD->u[faceID + 1]), 2)
                     + pow(0.5 * (m_WGD->v[faceID] + m_WGD->v[faceID + nx]), 2)
                     + pow(0.5 * (m_WGD->w[faceID] + m_WGD->w[faceID + nx * ny]), 2));
        nVal++;
      }
    }
  }
  uRef = uSum / nVal;

  uStar = 0.4 * uRef / log((zRef - d0d) / z0d);
  std::cout << "\t\t Mean reference velocity at zRef = " << zRef << " m ==> uRef = " << uRef << " m/s" << std::endl;
  std::cout << "\t\t Mean friction velocity uStar = " << uStar << " m/s" << std::endl;
}

void TURBGeneralData::getDerivatives()
{

  for (auto id = 0u; id < icellfluid.size(); id++) {
    int cellID = icellfluid[id];
    // linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //  i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    /*
      Diagonal componants of the strain-rate tensor naturally fall at
      the cell-center
    */

    // index of neighbour cells
    int faceID = i + j * nx + k * nx * ny;
    int idxp = faceID + 1;// i+1,j,k
    int idyp = faceID + nx;// i,j+1,k
    int idzp = faceID + ny * nx;// i,j,k+1

    // Sxx = dudx
    Sxx[cellID] = (m_WGD->u[idxp] - m_WGD->u[faceID]) / (x_fc[i + 1] - x_fc[i]);
    // Syy = dvdy
    Syy[cellID] = (m_WGD->v[idyp] - m_WGD->v[faceID]) / (y_fc[j + 1] - y_fc[j]);
    // Szz = dwdz
    Szz[cellID] = (m_WGD->w[idzp] - m_WGD->w[faceID]) / (z_fc[k + 1] - z_fc[k]);

    /*
      Off-diagonal componants of the strain-rate tensor require extra interpolation
      of the velocity field to get the derivative at the cell-center
    */

    // index of neighbour cells
    int idp, idm;
    // interpolated velocity field at neighbour cell center
    float up, um, vp, vm, wp, wm;

    //--------------------------------------
    // Sxy = 0.5*(dudy+dvdx) at z_cc
    // u_hat+
    idp = faceID + 1 + nx;// i+1,j+1
    idm = faceID + nx;// i,j+1
    up = ((x_cc[i] - x_fc[i]) * m_WGD->u[idp] + (x_fc[i + 1] - x_cc[i]) * m_WGD->u[idm]) / (x_fc[i + 1] - x_fc[i]);

    // u_hat-
    idp = faceID + 1 - nx;// i+1,j-1
    idm = faceID - nx;// i,j-1
    um = ((x_cc[i] - x_fc[i]) * m_WGD->u[idp] + (x_fc[i + 1] - x_cc[i]) * m_WGD->u[idm]) / (x_fc[i + 1] - x_fc[i]);

    // v_hat+
    idp = faceID + 1 + nx;// i+1,j+1
    idm = faceID + 1;// i+1,j
    vp = ((y_cc[j] - y_fc[j]) * m_WGD->v[idp] + (y_fc[j + 1] - y_cc[j]) * m_WGD->v[idm]) / (y_fc[j + 1] - y_fc[j]);

    // v_hat-
    idp = faceID - 1 + nx;// i-1,j+1
    idm = faceID - 1;// i-1,j
    vm = ((y_cc[j] - y_fc[j]) * m_WGD->v[idp] + (y_fc[j + 1] - y_cc[j]) * m_WGD->v[idm]) / (y_fc[j + 1] - y_fc[j]);

    // Sxy = 0.5*(dudy+dvdx) at z_cc
    Sxy[cellID] = 0.5 * ((up - um) / (y_cc[j + 1] - y_cc[j - 1]) + (vp - vm) / (x_cc[i + 1] - x_cc[i - 1]));

    //--------------------------------------
    // Sxz = 0.5*(dudz+dwdx) at y_cc
    // u_hat+
    idp = faceID + 1 + nx * ny;// i+1,k+1
    idm = faceID + nx * ny;// i,k+1
    up = ((x_cc[i] - x_fc[i]) * m_WGD->u[idp] + (x_fc[i + 1] - x_cc[i]) * m_WGD->u[idm]) / (x_fc[i + 1] - x_fc[i]);

    // u_hat-
    idp = faceID + 1 - nx * ny;// i+1,k-1
    idm = faceID - nx * ny;// i,k-1
    um = ((x_cc[i] - x_fc[i]) * m_WGD->u[idp] + (x_fc[i + 1] - x_cc[i]) * m_WGD->u[idm]) / (x_fc[i + 1] - x_fc[i]);

    // w_hat+
    idp = faceID + 1 + nx * ny;// i+1,k+1
    idm = faceID + 1;// i+1,k
    wp = ((z_cc[k] - z_fc[k]) * m_WGD->w[idp] + (z_fc[k + 1] - z_cc[k]) * m_WGD->w[idm]) / (z_fc[k + 1] - z_fc[k]);

    // w_hat-
    idp = faceID - 1 + nx * ny;// i-1,k+1
    idm = faceID - 1;// i-1,k
    wm = ((z_cc[k] - z_fc[k]) * m_WGD->w[idp] + (z_fc[k + 1] - z_cc[k]) * m_WGD->w[idm]) / (z_fc[k + 1] - z_fc[k]);

    // Sxz = 0.5*(dudz+dwdx) at y_cc
    Sxz[cellID] = 0.5 * ((up - um) / (z_cc[k + 1] - z_cc[k - 1]) + (wp - wm) / (x_cc[i + 1] - x_cc[i - 1]));

    //--------------------------------------
    // Syz = 0.5*(dvdz+dwdy) at x_cc
    // v_hat+
    idp = faceID + nx + nx * ny;// j+1,k+1
    idm = faceID + nx * ny;// j,k+1
    vp = ((y_cc[j] - y_fc[j]) * m_WGD->v[idp] + (y_fc[j + 1] - y_cc[j]) * m_WGD->v[idm]) / (y_fc[j + 1] - y_fc[j]);

    // v_hat-
    idp = faceID + nx - nx * ny;// j+1,k-1
    idm = faceID - nx * ny;// j,k-1
    vm = ((y_cc[j] - y_fc[j]) * m_WGD->v[idp] + (y_fc[j + 1] - y_cc[j]) * m_WGD->v[idm]) / (y_fc[j + 1] - y_fc[j]);

    // w_hat+
    idp = faceID + nx + nx * ny;// j+1,k+1
    idm = faceID + nx;// j+1,k
    wp = ((z_cc[k - 1] - z_fc[k]) * m_WGD->w[idp] + (z_fc[k + 1] - z_cc[k]) * m_WGD->w[idm]) / (z_fc[k + 1] - z_fc[k]);

    // w_hat-
    idp = faceID - nx + nx * ny;// j-1,k+1
    idm = faceID - nx;// j-1,k
    wp = ((z_cc[k] - z_fc[k]) * m_WGD->w[idp] + (z_fc[k + 1] - z_cc[k]) * m_WGD->w[idm]) / (z_fc[k + 1] - z_fc[k]);

    // Syz = 0.5*(dvdz+dwdy) at x_cc
    Syz[cellID] = 0.5 * ((vp - vm) / (z_cc[k + 1] - z_cc[k - 1]) + (wp - wm) / (y_cc[j + 1] - y_cc[j - 1]));
  }
}

void TURBGeneralData::getStressTensor()
{
  int cellID;
  for (auto id = 0u; id < icellfluid.size(); id++) {
    cellID = icellfluid[id];

    float NU_T = 0.0;
    float TKE = 0.0;
    float LM = Lm[cellID];

    //
    float SijSij = Sxx[cellID] * Sxx[cellID] + Syy[cellID] * Syy[cellID] + Szz[cellID] * Szz[cellID]
                   + 2.0 * (Sxy[cellID] * Sxy[cellID] + Sxz[cellID] * Sxz[cellID] + Syz[cellID] * Syz[cellID]);

    NU_T = LM * LM * sqrt(2.0 * SijSij);
    TKE = pow((NU_T / (cPope * LM)), 2.0);
    tke[cellID] = TKE;
    CoEps[cellID] = 5.7 * pow(sqrt(TKE) * cPope, 3.0) / (LM);

    txx[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Sxx[cellID]);
    tyy[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Syy[cellID]);
    tzz[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Szz[cellID]);
    txy[cellID] = -2.0 * (NU_T * Sxy[cellID]);
    txz[cellID] = -2.0 * (NU_T * Sxz[cellID]);
    tyz[cellID] = -2.0 * (NU_T * Syz[cellID]);

    txx[cellID] = fabs(sigUConst * txx[cellID]);
    tyy[cellID] = fabs(sigVConst * tyy[cellID]);
    tzz[cellID] = fabs(sigWConst * tzz[cellID]);
  }
}

void TURBGeneralData::derivativeVelocity()
{
  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;

    // linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //  i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);
    int faceID = i + j * nx + k * nx * ny;

    /*
     - Diagonal componants of the velocity gradient tensor naturally fall at the cell-center
     - Off-diagonal componants of the  velocity gradient tensor require extra interpolation
       of the velocity field to get the derivative at the cell-center  
     - Derivative with respect to z need to be adjusted for non-uniform z-grid
    */

    // Gxx = dudx
    Gxx[cellID] = (m_WGD->u[faceID + 1] - m_WGD->u[faceID]) / (m_WGD->dx);
    // Gyx = dvdx
    Gyx[cellID] = ((m_WGD->v[faceID + 1] + m_WGD->v[faceID + 1 + nx])
                   - (m_WGD->v[faceID - 1] - m_WGD->v[faceID - 1 + nx]))
                  / (4.0 * m_WGD->dx);
    // Gzx = dwdx
    Gzx[cellID] = ((m_WGD->w[faceID + 1] + m_WGD->w[faceID + 1 + nx * ny])
                   - (m_WGD->w[faceID - 1] + m_WGD->w[faceID - 1 + nx * ny]))
                  / (4.0 * m_WGD->dx);

    // Gxy = dudy
    Gxy[cellID] = ((m_WGD->u[faceID + nx] + m_WGD->u[faceID + 1 + nx])
                   - (m_WGD->u[faceID - nx] + m_WGD->u[faceID + 1 - nx]))
                  / (4.0 * m_WGD->dy);
    // Gyy = dvdy
    Gyy[cellID] = (m_WGD->v[faceID + nx] - m_WGD->v[faceID]) / (m_WGD->dy);
    // Gzy = dwdy
    Gzy[cellID] = ((m_WGD->w[faceID + nx] + m_WGD->w[faceID + nx + nx * ny])
                   - (m_WGD->w[faceID - nx] + m_WGD->w[faceID - nx + nx * ny]))
                  / (4.0 * m_WGD->dy);


    if (flagUniformZGrid) {
      // Gxz = dudz
      Gxz[cellID] = ((m_WGD->u[faceID + nx * ny] + m_WGD->u[faceID + 1 + nx * ny])
                     - (m_WGD->u[faceID - nx * ny] + m_WGD->u[faceID + 1 - nx * ny]))
                    / (4.0 * m_WGD->dz);
      // Gyz = dvdz
      Gyz[cellID] = ((m_WGD->v[faceID + nx * ny] + m_WGD->v[faceID + nx + nx * ny])
                     - (m_WGD->v[faceID - nx * ny] + m_WGD->v[faceID + nx - nx * ny]))
                    / (4.0 * m_WGD->dz);
      // Gzz = dwdz
      Gzz[cellID] = (m_WGD->w[faceID + nx * ny] - m_WGD->w[faceID]) / (m_WGD->dz);
    } else {
      // Gxz = dudz
      Gxz[cellID] = (0.5 * (m_WGD->z[k] - m_WGD->z[k - 1]) / (m_WGD->z[k + 1] - m_WGD->z[k])
                       * ((m_WGD->u[faceID + nx * ny] + m_WGD->u[faceID + 1 + nx * ny])
                          - (m_WGD->u[faceID] + m_WGD->u[faceID + 1]))
                     + 0.5 * (m_WGD->z[k + 1] - m_WGD->z[k]) / (m_WGD->z[k] - m_WGD->z[k - 1])
                         * ((m_WGD->u[faceID] + m_WGD->u[faceID + 1])
                            - (m_WGD->u[faceID - nx * ny] + m_WGD->u[faceID + 1 - nx * ny])))
                    / (m_WGD->z[k + 1] - m_WGD->z[k - 1]);
      // Gyz = dvdz
      Gyz[cellID] = (0.5 * (m_WGD->z[k] - m_WGD->z[k - 1]) / (m_WGD->z[k + 1] - m_WGD->z[k])
                       * ((m_WGD->v[faceID + nx * ny] + m_WGD->v[faceID + nx + nx * ny])
                          - (m_WGD->v[faceID] + m_WGD->v[faceID + nx]))
                     + 0.5 * (m_WGD->z[k + 1] - m_WGD->z[k]) / (m_WGD->z[k] - m_WGD->z[k - 1])
                         * ((m_WGD->v[faceID] + m_WGD->v[faceID + nx])
                            - (m_WGD->v[faceID - nx * ny] + m_WGD->v[faceID + nx - nx * ny])))
                    / (m_WGD->z[k + 1] - m_WGD->z[k - 1]);
      // Gzz = dwdz
      Gzz[cellID] = (m_WGD->w[faceID + nx * ny] - m_WGD->w[faceID]) / (m_WGD->dz_array[k]);
    }
  }

  std::cout << "\t\t Imposing Wall BC (log law)..." << std::endl;
  for (auto i = 0u; i < wallVec.size(); i++) {
    wallVec.at(i)->setWallsVelocityDeriv(m_WGD, this);
  }
  // std::cout<<"\t\t Wall BC done."<<std::endl;

  return;
}

void TURBGeneralData::stressTensor()
{
  float tkeBound = turbUpperBound * uStar * uStar;

  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;

    float Sxx = Gxx[cellID];
    float Syy = Gyy[cellID];
    float Szz = Gzz[cellID];
    float Sxy = 0.5 * (Gxy[cellID] + Gyx[cellID]);
    float Sxz = 0.5 * (Gxz[cellID] + Gzx[cellID]);
    float Syz = 0.5 * (Gyz[cellID] + Gzy[cellID]);

    float NU_T = 0.0;
    float TKE = 0.0;
    float LM = Lm[cellID];

    //
    float SijSij = Sxx * Sxx + Syy * Syy + Szz * Szz + 2.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);

    NU_T = LM * LM * sqrt(2.0 * SijSij);
    TKE = pow((NU_T / (cPope * LM)), 2.0);

    if (TKE > tkeBound)
      TKE = tkeBound;

    CoEps[cellID] = 5.7 * pow(sqrt(TKE) * cPope, 3.0) / (LM);
    tke[cellID] = TKE;

    txx[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Sxx);
    tyy[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Syy);
    tzz[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Szz);
    txy[cellID] = -2.0 * (NU_T * Sxy);
    txz[cellID] = -2.0 * (NU_T * Sxz);
    tyz[cellID] = -2.0 * (NU_T * Syz);

    txx[cellID] = fabs(sigUConst * txx[cellID]);
    tyy[cellID] = fabs(sigVConst * tyy[cellID]);
    tzz[cellID] = fabs(sigWConst * tzz[cellID]);
  }
}

void TURBGeneralData::addBackgroundMixing()
{
  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;

    txx[cellID] += backgroundMixing;
    tyy[cellID] += backgroundMixing;
    tzz[cellID] += backgroundMixing;
  }
  return;
}

void TURBGeneralData::divergenceStress()
{
  std::cout << "\t\t Computing Divergence of Stess Tensor..." << std::endl;

  // x-comp of the divergence of the stress tensor
  // div(tau)_x = dtxxdx + dtxydy + dtxzdz
  derivativeStress(txx, txy, txz, div_tau_x);

  // y-comp of the divergence of the stress tensor
  // div(tau)_y = dtxzdx + dtyydy + dtyzdz
  derivativeStress(txy, tyy, tyz, div_tau_y);

  // z-comp of the divergence of the stress tensor
  // div(tau)_z = dtxzdx + dtyzdy + dtzzdz
  derivativeStress(txz, tyz, tzz, div_tau_z);

  return;
}

void TURBGeneralData::derivativeStress(const std::vector<float> &tox,
                                       const std::vector<float> &toy,
                                       const std::vector<float> &toz,
                                       std::vector<float> &div_tau_o)
{
  // o-comp of the divergence of the stress tensor
  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;

    // linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //  i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));

    tmp_dtoxdx[cellID] = (tox[cellID + 1] - tox[cellID - 1]) / (2.0 * m_WGD->dx);
    tmp_dtoydy[cellID] = (toy[cellID + (nx - 1)] - toy[cellID - (nx - 1)]) / (2.0 * m_WGD->dy);

    if (flagUniformZGrid) {
      tmp_dtozdz[cellID] = (toz[cellID + (nx - 1) * (ny - 1)] - toz[cellID - (nx - 1) * (ny - 1)])
                           / (2.0 * m_WGD->dz);
    } else {
      tmp_dtozdz[cellID] = ((m_WGD->z[k] - m_WGD->z[k - 1]) / (m_WGD->z[k + 1] - m_WGD->z[k])
                              * (toz[cellID + (nx - 1) * (ny - 1)] - toz[cellID])
                            + (m_WGD->z[k + 1] - m_WGD->z[k]) / (m_WGD->z[k] - m_WGD->z[k - 1])
                                * (toz[cellID] - toz[cellID - (nx - 1) * (ny - 1)]))
                           / (m_WGD->z[k + 1] - m_WGD->z[k - 1]);
    }
  }
  // correction at the wall
  for (auto i = 0u; i < wallVec.size(); i++) {
    wallVec.at(i)->setWallsStressDeriv(m_WGD, this, tox, toy, toz);
  }
  // compute the the divergence
  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;
    div_tau_o[cellID] = tmp_dtoxdx[cellID] + tmp_dtoydy[cellID] + tmp_dtozdz[cellID];
  }

  return;
}

void TURBGeneralData::boundTurbFields()
{
  float stressBound = turbUpperBound * uStar * uStar;
  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;

    if (txx[cellID] < -stressBound)
      txx[cellID] = -stressBound;
    if (txx[cellID] > stressBound)
      txx[cellID] = stressBound;

    if (txy[cellID] < -stressBound)
      txy[cellID] = -stressBound;
    if (txy[cellID] > stressBound)
      txy[cellID] = stressBound;

    if (txz[cellID] < -stressBound)
      txz[cellID] = -stressBound;
    if (txz[cellID] > stressBound)
      txz[cellID] = stressBound;

    if (tyy[cellID] < -stressBound)
      tyy[cellID] = -stressBound;
    if (tyy[cellID] > stressBound)
      tyy[cellID] = stressBound;

    if (tyz[cellID] < -stressBound)
      tyz[cellID] = -stressBound;
    if (tyz[cellID] > stressBound)
      tyz[cellID] = stressBound;

    if (tzz[cellID] < -stressBound)
      tzz[cellID] = -stressBound;
    if (tzz[cellID] > stressBound)
      tzz[cellID] = stressBound;
  }
}
