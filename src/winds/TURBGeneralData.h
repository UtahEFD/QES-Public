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

  // General QUIC Domain Data
  ///@{
  /** number of cells */
  int nx, ny, nz;
  ///@}

  long numcell_cent; /**< Total number of cell-centered values in domain */
  long numcell_face; /**< Total number of face-centered values in domain */

  //nt - number of time instance in data
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

  std::vector<float> tke; /**< turbulence kinetic energy */
  std::vector<float> CoEps; /**< dissipation rate */

  LocalMixing *localMixing; /**< mixing length class */
  std::vector<double> mixingLengths; /**< distance to the wall */

protected:
  WINDSGeneralData *m_WGD;

  void getDerivativesGPU();

  void derivativeVelocity();

  void stressTensor();

  void divergenceStress();
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
  const float cPope = 0.55;
  float sigUOrg = 2.5;
  float sigVOrg = 2.0;
  float sigWOrg = 1.3;
  float sigUConst = 1.5 * sigUOrg * sigUOrg * cPope * cPope;
  float sigVConst = 1.5 * sigVOrg * sigVOrg * cPope * cPope;
  float sigWConst = 1.5 * sigWOrg * sigWOrg * cPope * cPope;
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