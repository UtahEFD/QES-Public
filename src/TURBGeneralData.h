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

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class TURBGeneralData {

public:
    TURBGeneralData()
    {}
    TURBGeneralData(const WINDSInputData*,WINDSGeneralData*);
    virtual ~TURBGeneralData()
    {}

    virtual void run(WINDSGeneralData*);

    bool flagNonLocalMixing;

    // General QUIC Domain Data
    int nx, ny, nz;		/**< number of cells */

    //grid information
    std::vector<float> x_fc;
    std::vector<float> x_cc;
    std::vector<float> y_fc;
    std::vector<float> y_cc;
    std::vector<float> z_fc;
    std::vector<float> z_cc;

    // Mean trubulence quantities
    float z0d,d0d;
    float zRef,uRef,uStar;
    float bldgH_mean,bldgH_max;
    float terrainH_max;

    // Turbulence Fields Upper Bound (tij < turbUpperBound*uStar^2)
    float turbUpperBound;

    // index for fluid cell
    std::vector<int> icellfluid;
    std::vector<int> iturbflag;
    /*
      0 - solid object, 1 - fluid
      2 - stairstep terrain-wall, 3 - cut-cell terrain
      4 - stairstep building-wall, 5 - cut-cell building
    */

    //strain rate tensor
    std::vector<float> Sxx;
    std::vector<float> Sxy;
    std::vector<float> Sxz;
    std::vector<float> Syy;
    std::vector<float> Syz;
    std::vector<float> Szz;

    //mixing length
    std::vector<float> Lm;

    // stress stensor
    std::vector<float> txx;
    std::vector<float> txy;
    std::vector<float> txz;
    std::vector<float> tyy;
    std::vector<float> tyz;
    std::vector<float> tzz;

    // derived turbulence quantities
    std::vector<float> tke;
    std::vector<float> CoEps;

    // local Mixing class and data
    LocalMixing* localMixing;
    std::vector<double> mixingLengths;

protected:

private:
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

    void getFrictionVelocity(WINDSGeneralData*);
    void getDerivatives(WINDSGeneralData*);
    void getStressTensor();
    
    void boundTurbFields();

};
