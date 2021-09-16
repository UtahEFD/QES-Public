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
  TURBGeneralData()
  {}
  TURBGeneralData(const WINDSInputData *, WINDSGeneralData *);
  TURBGeneralData(const std::string, WINDSGeneralData *);

  virtual ~TURBGeneralData()
  {}

  virtual void run(WINDSGeneralData *);

  // load data at given time instance
  void loadNetCDFData(int);

  bool flagUniformZGrid = true; /**< :document this: */
  bool flagNonLocalMixing; /**< :document this: */

  // General QUIC Domain Data
  ///@{
  /** number of cells */
  int nx, ny, nz;
  ///@}

  //nt - number of time instance in data
  int nt;
  // time vector
  std::vector<float> t;

  ///@{
  /** grid information */
  std::vector<float> x_fc;
  std::vector<float> x_cc;
  std::vector<float> y_fc;
  std::vector<float> y_cc;
  std::vector<float> z_fc;
  std::vector<float> z_cc;
  ///@}

  // Mean trubulence quantities
  float z0d, d0d;
  float zRef, uRef, uStar;
  float bldgH_mean, bldgH_max;
  float terrainH_max;

  // Turbulence Fields Upper Bound (tij < turbUpperBound*uStar^2)
  float turbUpperBound; /**< Turbulence fields upper bound */

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

  ///@{
  /** strain rate tensor */
  std::vector<float> Sxx;
  std::vector<float> Sxy;
  std::vector<float> Sxz;
  std::vector<float> Syy;
  std::vector<float> Syz;
  std::vector<float> Szz;
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

  // derived turbulence quantities
  std::vector<float> tke;
  std::vector<float> CoEps;

  // local Mixing class and data
  LocalMixing *localMixing;
  std::vector<double> mixingLengths;

protected:
private:
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

  // input: store here for multiple time instance.
  NetCDFInput *input;

  void getFrictionVelocity(WINDSGeneralData *);

  void getDerivatives(WINDSGeneralData *);
  void getDerivatives_v2(WINDSGeneralData *);

  void getStressTensor();
  void getStressTensor_v2();


  void boundTurbFields();
};
