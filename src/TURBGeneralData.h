#pragma once

#include <math.h>
#include <vector>
//#include "Args.hpp"
#include "URBGeneralData.h"
#include "TURBLocalMixing.h"
#include "TURBWall.h"
#include "TURBWallBuilding.h"
#include "TURBWallTerrain.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
 */

class TURBGeneralData {

public:
  TURBGeneralData()
  {}
  //TURBGeneralData(Args*,URBGeneralData*);
  TURBGeneralData(URBGeneralData*);
  virtual ~TURBGeneralData()
    {}

  virtual void run(URBGeneralData*);

  void getDerivatives(URBGeneralData*);
  void getStressTensor();

  // General QUIC Domain Data
  int nx, ny, nz;		/**< number of cells */
  float dx, dy, dz;		/**< Grid resolution*/

  //grid information
  std::vector<float> x_fc;
  std::vector<float> x_cc;
  std::vector<float> y_fc;
  std::vector<float> y_cc;
  std::vector<float> z_fc;
  std::vector<float> z_cc;

  // index for fluid cell
  std::vector<int> icellfluid;
  std::vector<int> iturbflag;
  /*
    0 - solid object, 1 - fluid 
    2 - stairstep terrain-wall, 3 - cut-cell terrain
    4 - stairstep building-wall, 5 - cut-cell building 
   */
  
  //strain rate tensor
  std::vector<float> S11;
  std::vector<float> S12;
  std::vector<float> S13;
  std::vector<float> S22;
  std::vector<float> S23;
  std::vector<float> S33;

  //mixing length
  std::vector<float> Lm;

  // stress stensor
  std::vector<float> tau11;
  std::vector<float> tau12;
  std::vector<float> tau13;
  std::vector<float> tau22;
  std::vector<float> tau23;
  std::vector<float> tau33;

  // derived turbulence quantities
  std::vector<float> tke;
  std::vector<float> CoEps; 
  
 protected:

 private:
  // FM -> obsolete
  TURBLocalMixing *LocalMixing;

  // store the wall classes
  std::vector<TURBWall *>wallVec;

   // some constants for turbulent model
  const float vonKar=0.41;
  const float cPope=0.55;
  const float sigUOrg= 1.8;
  const float sigVOrg=2.0;
  const float sigWOrg=1.3;
  const float sigUConst=sigUOrg*sigUOrg*cPope*cPope;//2.3438;
  const float sigVConst=sigVOrg*sigVOrg*cPope*cPope;//1.5;
  const float sigWConst=sigWOrg*sigWOrg*cPope*cPope;//0.6338;
  
};
