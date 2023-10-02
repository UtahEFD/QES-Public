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
  std::cout << "-------------------------------------------------------------------" << std::endl;
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
    std::cout << "[QES-TURB]\t Non-Local mixing for buidlings: ON \n";
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

  numcell_cent = m_WGD->numcell_cent;

  dz = m_WGD->dz;
  dy = m_WGD->dy;
  dx = m_WGD->dx;

  // x-grid (cell-center)
  x.resize(nx - 1, 0);
  x = m_WGD->x;

  // y-grid (cell-center)
  y.resize(ny - 1, 0);
  y = m_WGD->y;

  // z-grid (face-center & cell-center)
  // z cell-center
  z.resize(nz - 1, 0);
  z = m_WGD->z;
  // z face-center (no ghost cell under the ground)
  z_face.resize(nz, 0);
  z_face = m_WGD->z_face;
  // dz_array
  dz_array.resize(nz - 1, 0);
  dz_array = m_WGD->dz_array;

  iturbflag.resize(numcell_cent, 0);

  /*
     vector containing cell id of fluid cell
     do not include 1 cell shell around the domain
     => i=1...nx-2 j=1...ny-2
     do not include 1 cell layer at the top of the domain
     => k=1...nz-2
  */
  for (int k = 1; k < nz - 2; k++) {
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
  std::cout << "[QES-TURB]\t Defining Solid Walls...\n";
  wallVec.push_back(new TURBWallBuilding(WID, m_WGD, this));
  wallVec.push_back(new TURBWallTerrain(WID, m_WGD, this));
  // std::cout << "\t\t Walls Defined...\n";

  // mixing length

  std::cout << "[QES-TURB]\t Defining Local Mixing Length...\n";
  auto mlStartTime = std::chrono::high_resolution_clock::now();
  if (WID->turbParams->methodLocalMixing == 0) {
    std::cout << "[QES-TURB]\t Default Local Mixing Length...\n";
    localMixing = new LocalMixingDefault();
  } else if (WID->turbParams->methodLocalMixing == 1) {
    std::cout << "[QES-TURB]\t Computing Local Mixing Length using serial code...\n";
    localMixing = new LocalMixingSerial();
  } else if (WID->turbParams->methodLocalMixing == 2) {
    /*******Add raytrace code here********/
    std::cout << "[QES-TURB]\t Computing mixing length scales..." << std::endl;
    // WID->simParams->DTE_mesh->calculateMixingLength(nx, ny, nz, dx, dy, dz, WGD->icellflag, WGD->mixingLengths);
  } else if (WID->turbParams->methodLocalMixing == 3) {
    localMixing = new LocalMixingOptix();
  } else if (WID->turbParams->methodLocalMixing == 4) {
    std::cout << "[QES-TURB]\t Loading Local Mixing Length data form NetCDF...\n";
    localMixing = new LocalMixingNetCDF();
  } else {
    // this should not happen (checked in TURBParams)
  }

  localMixing->defineMixingLength(WID, m_WGD);

  Lm.resize(numcell_cent, 0.0);
  // make a copy as mixing length will be modifiy by non local
  // (need to be reset at each time instances)
  for (auto id = 0u; id < icellfluid.size(); id++) {
    int cellId = icellfluid[id];
    Lm[cellId] = vonKar * m_WGD->mixingLengths[cellId];
  }

  auto mlEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> mlElapsed = mlEndTime - mlStartTime;
  std::cout << "[QES-TURB]\t Local Mixing Defined: elapsed time: " << mlElapsed.count() << " s\n";

  // Caluclating Turbulence quantities (u*,z0,d0) based on morphology of domain.
  bldgH_mean = 0.0;
  bldgH_max = 0.0;
  terrainH_max = 0.0;
  zRef = 0.0;
  uRef = 0.0;
  uStar = 0.0;

  std::cout << "[QES-TURB]\t Calculating Morphometric parametrization of trubulence..." << std::endl;

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

  std::cout << "[QES-TURB]\t Computing friction velocity..." << std::endl;
  frictionVelocity();

  std::cout << "[QES-TURB]\t Allocating memory...\n";

  // comp. of the velocity gradient tensor
  Gxx.resize(numcell_cent, 0);
  Gxy.resize(numcell_cent, 0);
  Gxz.resize(numcell_cent, 0);
  Gyx.resize(numcell_cent, 0);
  Gyy.resize(numcell_cent, 0);
  Gyz.resize(numcell_cent, 0);
  Gzx.resize(numcell_cent, 0);
  Gzy.resize(numcell_cent, 0);
  Gzz.resize(numcell_cent, 0);

  // comp of the stress tensor
  txx.resize(numcell_cent, 0);
  txy.resize(numcell_cent, 0);
  txz.resize(numcell_cent, 0);
  tyy.resize(numcell_cent, 0);
  tyz.resize(numcell_cent, 0);
  tzz.resize(numcell_cent, 0);

  // derived turbulence quantities
  tke.resize(numcell_cent, 0);
  CoEps.resize(numcell_cent, 0);
  nuT.resize(numcell_cent, 0);
  // std::cout << "\t\t Memory allocation completed.\n";

  if (flagCompDivStress) {
    // comp of the divergence of the stress tensor
    tmp_dtoxdx.resize(numcell_cent, 0);
    tmp_dtoydy.resize(numcell_cent, 0);
    tmp_dtozdz.resize(numcell_cent, 0);

    // comp of the divergence of the stress tensor
    div_tau_x.resize(numcell_cent, 0);
    div_tau_y.resize(numcell_cent, 0);
    div_tau_z.resize(numcell_cent, 0);
  }

  auto EndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> Elapsed = EndTime - StartTime;

  std::cout << "[QES-TURB]\t Initialization of turbulence model completed.\n";
  std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << endl;
}

TURBGeneralData::TURBGeneralData(const std::string inputFile, WINDSGeneralData *WGDin)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
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

  // get time variables
  t.resize(nt);
  input->getVariableData("t", t);

  // check if times is in the NetCDF file
  NcVar NcVar_timestamp;
  input->getVariable("timestamp", NcVar_timestamp);

  if (NcVar_timestamp.isNull()) {
    QESout::warning("No timestamp in NetCDF file");
  }

  // make local copy of grid information
  // nx,ny,nz consitant with QES-Winds (face-center)
  nx = m_WGD->nx;
  ny = m_WGD->ny;
  nz = m_WGD->nz;

  numcell_cent = m_WGD->numcell_cent;
  numcell_face = m_WGD->numcell_face;

  dx = m_WGD->dx;
  dy = m_WGD->dy;
  dz = m_WGD->dz;

  // x-grid (cell-center)
  x.resize(nx - 1, 0);
  x = m_WGD->x;

  // y-grid (cell-center)
  y.resize(ny - 1, 0);
  y = m_WGD->y;

  // z-grid (face-center & cell-center)
  // z cell-center
  z.resize(nz - 1, 0);
  z = m_WGD->z;
  // z face-center (no ghost cell under the ground)
  z_face.resize(nz, 0);
  z_face = m_WGD->z_face;
  // dz_array
  dz_array.resize(nz - 1, 0);
  dz_array = m_WGD->dz_array;

  /*
     vector containing cell id of fluid cell
     do not include 1 cell shell around the domain
     => i=1...nx-2 j=1...ny-2
     do not include 1 cell layer at the top of the domain
     => k=1...nz-2
  */
  for (int k = 1; k < nz - 2; k++) {
    for (int j = 1; j < ny - 2; j++) {
      for (int i = 1; i < nx - 2; i++) {
        int id = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (m_WGD->icellflag[id] != 0 && m_WGD->icellflag[id] != 2) {
          icellfluid.push_back(id);
        }
      }
    }
  }
  // comp. of the velocity gradient tensor
  Gxx.resize(numcell_cent, 0);
  Gxy.resize(numcell_cent, 0);
  Gxz.resize(numcell_cent, 0);
  Gyx.resize(numcell_cent, 0);
  Gyy.resize(numcell_cent, 0);
  Gyz.resize(numcell_cent, 0);
  Gzx.resize(numcell_cent, 0);
  Gzy.resize(numcell_cent, 0);
  Gzz.resize(numcell_cent, 0);

  // comp of the stress tensor
  txx.resize(numcell_cent, 0);
  txy.resize(numcell_cent, 0);
  txz.resize(numcell_cent, 0);
  tyy.resize(numcell_cent, 0);
  tyz.resize(numcell_cent, 0);
  tzz.resize(numcell_cent, 0);

  // derived turbulence quantities
  tke.resize(numcell_cent, 0);
  CoEps.resize(numcell_cent, 0);
  nuT.resize(numcell_cent, 0);

  // comp of the divergence of the stress tensor
  tmp_dtoxdx.resize(numcell_cent, 0);
  tmp_dtoydy.resize(numcell_cent, 0);
  tmp_dtozdz.resize(numcell_cent, 0);

  // comp of the divergence of the stress tensor
  div_tau_x.resize(numcell_cent, 0);
  div_tau_y.resize(numcell_cent, 0);
  div_tau_z.resize(numcell_cent, 0);
}

TURBGeneralData::TURBGeneralData(WINDSGeneralData *WGDin)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[QES-TURB]\t Initialization of turbulence model...\n";
  m_WGD = WGDin;
  // make local copy of grid information
  // nx,ny,nz consitant with WINDS (face-center)
  // WINDS->grid correspond to face-center grid
  nz = m_WGD->nz;
  ny = m_WGD->ny;
  nx = m_WGD->nx;

  numcell_cent = m_WGD->numcell_cent;
  numcell_face = m_WGD->numcell_face;

  dz = m_WGD->dz;
  dy = m_WGD->dy;
  dx = m_WGD->dx;

  // x-grid (cell-center)
  x.resize(nx - 1, 0);
  x = m_WGD->x;

  // y-grid (cell-center)
  y.resize(ny - 1, 0);
  y = m_WGD->y;

  // z-grid (face-center & cell-center)
  // z cell-center
  z.resize(nz - 1, 0);
  z = m_WGD->z;
  // z face-center (no ghost cell under the ground)
  z_face.resize(nz - 1, 0);
  z_face = m_WGD->z_face;

  /*
     vector containing cell id of fluid cell
     do not include 1 cell shell around the domain
     => i=1...nx-2 j=1...ny-2
     do not include 1 cell layer at the top of the domain
     => k=1...nz-2

  */
  /*
  for (int k = 1; k < nz - 2; k++) {
    for (int j = 1; j < ny - 2; j++) {
      for (int i = 1; i < nx - 2; i++) {
        int id = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (m_WGD->icellflag[id] != 0 && m_WGD->icellflag[id] != 2) {
          icellfluid.push_back(id);
        }
      }
    }
  }
  */
  for (int k = 1; k < nz - 2; k++) {
    for (int j = 1; j < ny - 2; j++) {
      for (int i = 1; i < nx - 2; i++) {
        int id = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (m_WGD->icellflag[id] != 0 && m_WGD->icellflag[id] != 2) {
          icellfluid.push_back(id);
        }
      }
    }
  }
  // comp. of the velocity gradient tensor
  Gxx.resize(numcell_cent, 0);
  Gxy.resize(numcell_cent, 0);
  Gxz.resize(numcell_cent, 0);
  Gyx.resize(numcell_cent, 0);
  Gyy.resize(numcell_cent, 0);
  Gyz.resize(numcell_cent, 0);
  Gzx.resize(numcell_cent, 0);
  Gzy.resize(numcell_cent, 0);
  Gzz.resize(numcell_cent, 0);

  // comp of the stress tensor
  txx.resize(numcell_cent, 0);
  txy.resize(numcell_cent, 0);
  txz.resize(numcell_cent, 0);
  tyy.resize(numcell_cent, 0);
  tyz.resize(numcell_cent, 0);
  tzz.resize(numcell_cent, 0);

  // derived turbulence quantities
  tke.resize(numcell_cent, 0);
  CoEps.resize(numcell_cent, 0);
  nuT.resize(numcell_cent, 0);

  // comp of the divergence of the stress tensor
  tmp_dtoxdx.resize(numcell_cent, 0);
  tmp_dtoydy.resize(numcell_cent, 0);
  tmp_dtozdz.resize(numcell_cent, 0);

  // comp of the divergence of the stress tensor
  div_tau_x.resize(numcell_cent, 0);
  div_tau_y.resize(numcell_cent, 0);
  div_tau_z.resize(numcell_cent, 0);
}

void TURBGeneralData::loadNetCDFData(int stepin)
{
  std::cout << "[QES-TURB]\t loading data at step " << stepin << std::endl;

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
  // std::cout << "\t\t Memory allocation completed.\n";
  std::cout << "[QES-TURB]\t Initialization of turbulence model completed.\n";
}

// compute turbulence fields
void TURBGeneralData::run()
{

  auto StartTime = std::chrono::high_resolution_clock::now();
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[QES-TURB]\t Running turbulence model..." << std::endl;

  std::cout << "[QES-TURB]\t Computing friction velocity..." << std::endl;
  frictionVelocity();

  std::cout << "[QES-TURB]\t Computing Derivatives (Strain Rate)..." << std::endl;
  derivativeVelocity();
  // std::cout<<"\t\t Derivatives computed."<<std::endl;

  std::cout << "[QES-TURB]\t Computing local mixing length model..." << std::endl;
  getTurbulentViscosity();

  //  if (m_WGD->canopy) {
  //    std::cout << "Applying canopy wake turbulence parameterization...\n";
  //    m_WGD->canopy->applyCanopyTurbulenceWake(m_WGD, this);
  //  }
  if (m_WGD->canopy) {
    std::cout << "[QES-TURB]\t Applying canopy wake turbulence parameterization...\n";
    m_WGD->canopy->applyCanopyTurbulenceWake(m_WGD, this);
  }

  std::cout << "[QES-TURB]\t Computing Stess Tensor..." << std::endl;
  stressTensor();
  // std::cout<<"\t\t Stress Tensor computed."<<std::endl;

  if (m_WGD->canopy) {
    std::cout << "[QES-TURB]\t Applying canopy stresses...\n";
    m_WGD->canopy->applyCanopyStress(m_WGD, this);
  }

  if (flagNonLocalMixing) {
    std::cout << "[QES-TURB]\t Applying non-local mixing..." << std::endl;

    for (size_t i = 0; i < m_WGD->allBuildingsV.size(); i++) {
      m_WGD->allBuildingsV[m_WGD->building_id[i]]->NonLocalMixing(m_WGD, this, m_WGD->building_id[i]);
    }
    // std::cout<<"\t\t Non-local mixing completed."<<std::endl;
  }

  std::cout << "[QES-TURB]\t Checking Upper Bound of Turbulence Fields..." << std::endl;
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
      std::vector<float>::iterator itr = std::lower_bound(z_face.begin(), z_face.end(), zRef);
      int k;
      if (itr != z_face.end()) {
        k = itr - z_face.begin();
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
  std::cout << "[QES-TURB]\t Mean reference velocity at zRef = " << zRef << " m ==> uRef = " << uRef << " m/s" << std::endl;
  std::cout << "[QES-TURB]\t Mean friction velocity uStar = " << uStar << " m/s" << std::endl;
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


    // - Diagonal componants of the velocity gradient tensor naturally fall at the cell-center
    // - Off-diagonal componants of the  velocity gradient tensor require extra interpolation
    //   of the velocity field to get the derivative at the cell-center
    // - Derivative with respect to z need to be adjusted for non-uniform z-grid


    // Gxx = dudx
    Gxx[cellID] = (m_WGD->u[faceID + 1] - m_WGD->u[faceID]) / (m_WGD->dx);
    // Gyx = dvdx
    Gyx[cellID] = ((m_WGD->v[faceID + 1] + m_WGD->v[faceID + 1 + nx])
                   - (m_WGD->v[faceID - 1] + m_WGD->v[faceID - 1 + nx]))
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

  std::cout << "[QES-TURB]\t Imposing Wall BC (log law)..." << std::endl;
  for (auto i = 0u; i < wallVec.size(); i++) {
    wallVec.at(i)->setWallsVelocityDeriv(m_WGD, this);
  }
  // std::cout<<"\t\t Wall BC done."<<std::endl;

  return;
}

void TURBGeneralData::getTurbulentViscosity()
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

    nuT[cellID] = NU_T;
    CoEps[cellID] = 5.7 * pow(sqrt(TKE) * cPope, 3.0) / (LM);
    tke[cellID] = TKE;
  }
  return;
}

void TURBGeneralData::stressTensor()
{
  float tkeBound = turbUpperBound * uStar * uStar;

  double R11, R12, R13, R21, R22, R23, R31, R32, R33;// rotation matrix
  double I11, I12, I13, I21, I22, I23, I31, I32, I33;// inverse of rotation matrix
  double P11, P12, P13, P21, P22, P23, P31, P32, P33;// P = tau*inv(R)

  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);
    int faceID = i + j * nx + k * nx * ny;

    float Sxx = Gxx[cellID];
    float Syy = Gyy[cellID];
    float Szz = Gzz[cellID];
    float Sxy = 0.5 * (Gxy[cellID] + Gyx[cellID]);
    float Sxz = 0.5 * (Gxz[cellID] + Gzx[cellID]);
    float Syz = 0.5 * (Gyz[cellID] + Gzy[cellID]);

    // float NU_T = 0.0;
    // float TKE = 0.0;
    float NU_T = nuT[cellID];
    float TKE = tke[cellID];
    float LM = Lm[cellID];

    //
    float SijSij = Sxx * Sxx + Syy * Syy + Szz * Szz + 2.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);

    // NU_T = LM * LM * sqrt(2.0 * SijSij);
    // TKE = pow((NU_T / (cPope * LM)), 2.0);

    if (TKE > tkeBound)
      TKE = tkeBound;

    CoEps[cellID] = 5.7 * pow(sqrt(TKE) * cPope, 3.0) / (LM);
    // tke[cellID] = TKE;

    txx[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Sxx);
    tyy[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Syy);
    tzz[cellID] = (2.0 / 3.0) * TKE - 2.0 * (NU_T * Szz);
    txy[cellID] = -2.0 * (NU_T * Sxy);
    txz[cellID] = -2.0 * (NU_T * Sxz);
    tyz[cellID] = -2.0 * (NU_T * Syz);

    // ROTATE INTO SENSOR-ALIGNED

    // float sensorDir = WID->metParams->sensors[i]->TS[0]->site_wind_dir[0];

    int k_ref = 0;
    float refHeight = 15.0;// reference height for rotation wind direction is arbitrarily set to 15m above terrain
    while (m_WGD->z_face[k_ref] < (refHeight + m_WGD->terrain[i + j * (m_WGD->nx - 1)])) {
      k_ref += 1;
    }

    int localRef = i + j * nx + k_ref * nx * ny;
    float dirRot = atan2(m_WGD->v[localRef], m_WGD->u[localRef]);// radians on the unit circle

    // Rotation matrix
    R11 = cos(dirRot);
    R12 = -sin(dirRot);
    R13 = 0.0;
    R21 = sin(dirRot);
    R22 = cos(dirRot);
    R23 = 0.0;
    R31 = 0.0;
    R32 = 0.0;
    R33 = 1.0;

    I11 = R11;
    I12 = R12;
    I13 = R13;
    I21 = R21;
    I22 = R22;
    I23 = R23;
    I31 = R31;
    I32 = R32;
    I33 = R33;

    double txx_temp = txx[cellID];
    double txy_temp = txy[cellID];
    double txz_temp = txz[cellID];
    double tyy_temp = tyy[cellID];
    double tyz_temp = tyz[cellID];
    double tzz_temp = tzz[cellID];

    // Invert rotation matrix
    invert3(I11, I12, I13, I21, I22, I23, I31, I32, I33);


    matMult(txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp, I11, I12, I13, I21, I22, I23, I31, I32, I33, P11, P12, P13, P21, P22, P23, P31, P32, P33);

    matMult(R11, R12, R13, R21, R22, R23, R31, R32, R33, P11, P12, P13, P21, P22, P23, P31, P32, P33, txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp);

    txx_temp = fabs(sigUConst * txx_temp);
    tyy_temp = fabs(sigVConst * tyy_temp);
    tzz_temp = fabs(sigWConst * tzz_temp);


    // DEROTATE
    // Rotation matrix
    dirRot = -dirRot;
    R11 = cos(dirRot);
    R12 = -sin(dirRot);
    R13 = 0.0;
    R21 = sin(dirRot);
    R22 = cos(dirRot);
    R23 = 0;
    R31 = 0;
    R32 = 0;
    R33 = 1;

    I11 = R11;
    I12 = R12;
    I13 = R13;
    I21 = R21;
    I22 = R22;
    I23 = R23;
    I31 = R31;
    I32 = R32;
    I33 = R33;

    // Invert rotation matrix
    invert3(I11, I12, I13, I21, I22, I23, I31, I32, I33);

    matMult(txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp, I11, I12, I13, I21, I22, I23, I31, I32, I33, P11, P12, P13, P21, P22, P23, P31, P32, P33);

    matMult(R11, R12, R13, R21, R22, R23, R31, R32, R33, P11, P12, P13, P21, P22, P23, P31, P32, P33, txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp);

    txx[cellID] = txx_temp;
    txy[cellID] = txy_temp;
    txz[cellID] = txz_temp;
    tyy[cellID] = tyy_temp;
    tyz[cellID] = tyz_temp;
    tzz[cellID] = tzz_temp;
  }
}

void TURBGeneralData::invert3(double &A_11,
                              double &A_12,
                              double &A_13,
                              double &A_21,
                              double &A_22,
                              double &A_23,
                              double &A_31,
                              double &A_32,
                              double &A_33)
{

  // calculate the determinant
  double det = A_11 * (A_22 * A_33 - A_23 * A_32) - A_12 * (A_21 * A_33 - A_23 * A_31) + A_13 * (A_21 * A_32 - A_22 * A_31);

  // calculate the inverse
  double Ainv_11 = (A_22 * A_33 - A_23 * A_32) / det;
  double Ainv_12 = -(A_12 * A_33 - A_13 * A_32) / det;
  double Ainv_13 = (A_12 * A_23 - A_22 * A_13) / det;
  double Ainv_21 = -(A_21 * A_33 - A_23 * A_31) / det;
  double Ainv_22 = (A_11 * A_33 - A_13 * A_31) / det;
  double Ainv_23 = -(A_11 * A_23 - A_13 * A_21) / det;
  double Ainv_31 = (A_21 * A_32 - A_31 * A_22) / det;
  double Ainv_32 = -(A_11 * A_32 - A_12 * A_31) / det;
  double Ainv_33 = (A_11 * A_22 - A_12 * A_21) / det;

  A_11 = Ainv_11;
  A_12 = Ainv_12;
  A_13 = Ainv_13;
  A_21 = Ainv_21;
  A_22 = Ainv_22;
  A_23 = Ainv_23;
  A_31 = Ainv_31;
  A_32 = Ainv_32;
  A_33 = Ainv_33;
}

void TURBGeneralData::matMult(const double &A11,
                              const double &A12,
                              const double &A13,
                              const double &A21,
                              const double &A22,
                              const double &A23,
                              const double &A31,
                              const double &A32,
                              const double &A33,
                              const double &B11,
                              const double &B12,
                              const double &B13,
                              const double &B21,
                              const double &B22,
                              const double &B23,
                              const double &B31,
                              const double &B32,
                              const double &B33,
                              double &C11,
                              double &C12,
                              double &C13,
                              double &C21,
                              double &C22,
                              double &C23,
                              double &C31,
                              double &C32,
                              double &C33)

{
  C11 = A11 * B11 + A12 * B21 + A13 * B31;
  C12 = A11 * B12 + A12 * B22 + A13 * B32;
  C13 = A11 * B13 + A12 * B23 + A13 * B33;
  C21 = A21 * B11 + A22 * B21 + A23 * B31;
  C22 = A21 * B12 + A22 * B22 + A23 * B32;
  C23 = A21 * B13 + A22 * B23 + A23 * B33;
  C31 = A31 * B11 + A32 * B21 + A33 * B31;
  C32 = A31 * B12 + A32 * B22 + A33 * B32;
  C33 = A31 * B13 + A32 * B23 + A33 * B33;
}


void TURBGeneralData::addBackgroundMixing()
{
  for (std::vector<int>::iterator it = icellfluid.begin(); it != icellfluid.end(); ++it) {
    int cellID = *it;

    txx[cellID] += backgroundMixing * backgroundMixing;
    tyy[cellID] += backgroundMixing * backgroundMixing;
    tzz[cellID] += backgroundMixing * backgroundMixing;
  }
  return;
}

void TURBGeneralData::divergenceStress()
{
  std::cout << "[QES-TURB]\t Computing Divergence of Stess Tensor..." << std::endl;

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
