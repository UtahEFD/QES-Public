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

/** @file TURBGeneralData.h */

#pragma once

#include <math.h>
#include <vector>

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "TURBWall.h"
#include "TURBWallBuilding.h"
#include "TURBWallTerrain.h"

#include "LocalMixing.h"
#include "LocalMixingDefault.h"
#include "LocalMixingNetCDF.h"
#include "LocalMixingSerial.h"
#include "LocalMixingOptix.h"


/**
 * @class TURBGeneralData
 * @brief :document this:
 */
class TURBGeneralData
{

public:
  TURBGeneralData(const WINDSInputData *, WINDSGeneralData *);
  TURBGeneralData(const std::string, WINDSGeneralData *);
  TURBGeneralData(WINDSGeneralData *);
  virtual ~TURBGeneralData()
  {}

  virtual void run();

  void loadNetCDFData(int);
  void divergenceStress();

  // General QUIC Domain Data
  ///@{
  /** number of cells */
  int nx, ny, nz;
  ///@}

  long numcell_cent; /**< Total number of cell-centered values in domain */
  long numcell_face; /**< Total number of face-centered values in domain */

  // nt - number of time instance in data
  int nt;
  // time vector
  std::vector<float> t;

  ///@{
  /** grid information */
  std::vector<float> x, y, z;

  std::vector<float> x_cc;
  std::vector<float> z_face;
  std::vector<float> dz_array;
  ///@}

  // index for fluid cell
  std::vector<int> icellfluid; /**< :document this: */
  std::vector<int> iturbflag; /**< :document this: */
  /*
    0 - solid object, 1 - fluid
    2 - stairstep terrain-wall, 3 - cut-cell terrain
    4 - stairstep building-wall, 5 - cut-cell building
  */

  ///@{
  /** Velocity gradient tensor */
  std::vector<float> Gxx;
  std::vector<float> Gxy;
  std::vector<float> Gxz;
  std::vector<float> Gyx;
  std::vector<float> Gyy;
  std::vector<float> Gyz;
  std::vector<float> Gzx;
  std::vector<float> Gzy;
  std::vector<float> Gzz;
  ///@}

  std::vector<float> Lm; /**< mixing length */

  ///@{
  /** stress tensor */
  std::vector<float> txx;
  std::vector<float> txy;
  std::vector<float> txz;
  std::vector<float> tyy;
  std::vector<float> tyz;
  std::vector<float> tzz;
  ///@}

  ///@{
  /** derivative of the stress */
  std::vector<float> tmp_dtoxdx;
  std::vector<float> tmp_dtoydy;
  std::vector<float> tmp_dtozdz;
  ///@}

  ///@{
  /** divergence of the stress */
  std::vector<float> div_tau_x;
  std::vector<float> div_tau_y;
  std::vector<float> div_tau_z;
  ///@}

  std::vector<float> nuT; /**< turbulent viscosity */
  std::vector<float> tke; /**< turbulence kinetic energy */
  std::vector<float> CoEps; /**< dissipation rate */

  LocalMixing *localMixing; /**< mixing length class */
  std::vector<double> mixingLengths; /**< distance to the wall */


  void invert3(double &A_11,
               double &A_12,
               double &A_13,
               double &A_21,
               double &A_22,
               double &A_23,
               double &A_31,
               double &A_32,
               double &A_33);

  void matMult(const double &A_11,
               const double &A_12,
               const double &A_13,
               const double &A_21,
               const double &A_22,
               const double &A_23,
               const double &A_31,
               const double &A_32,
               const double &A_33,
               const double &B_11,
               const double &B_12,
               const double &B_13,
               const double &B_21,
               const double &B_22,
               const double &B_23,
               const double &B_31,
               const double &B_32,
               const double &B_33,
               double &C_11,
               double &C_12,
               double &C_13,
               double &C_21,
               double &C_22,
               double &C_23,
               double &C_31,
               double &C_32,
               double &C_33);

  const float cPope = 0.55;
  float sigUOrg = 2.5;
  float sigVOrg = 2.0;
  float sigWOrg = 1.3;
  float sigUConst = 1.5 * sigUOrg * sigUOrg * cPope * cPope;
  float sigVConst = 1.5 * sigVOrg * sigVOrg * cPope * cPope;
  float sigWConst = 1.5 * sigWOrg * sigWOrg * cPope * cPope;

protected:
  WINDSGeneralData *m_WGD;

  void getDerivativesGPU();

  void derivativeVelocity();

  void getTurbulentViscosity();
  void stressTensor();

  void derivativeStress(const std::vector<float> &,
                        const std::vector<float> &,
                        const std::vector<float> &,
                        std::vector<float> &);

  void addBackgroundMixing();
  void frictionVelocity();
  void boundTurbFields();

private:
  // cannot have an empty constructor (have to pass in a mesh to build)
  TURBGeneralData();

  // store the wall classes
  std::vector<TURBWall *> wallVec;

  // some constants for turbulent model
  const float vonKar = 0.4;
  float backgroundMixing = 0.0;

  float dx, dy, dz;

  // input: store here for multiple time instance.
  NetCDFInput *input;

  bool flagUniformZGrid = true; /**< :document this: */
  bool flagNonLocalMixing; /**< :document this: */
  bool flagCompDivStress = true; /**< :document this: */

  // Mean trubulence quantities
  float z0d, d0d;
  float zRef, uRef, uStar;
  float bldgH_mean, bldgH_max;
  float terrainH_max;

  // Turbulence Fields Upper Bound (tij < turbUpperBound*uStar^2)
  float turbUpperBound; /**< Turbulence fields upper bound */
};
