#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "Canopy.h"

class CanopyROC : public CanopyElement
{
public:
  CanopyROC()
  {
  }

  // LDU: this gets run first, when the XML is parsed. So polygonVertices are always the rotated ones, wherever they're used downstream. However, the wake rotation seems to be done inside the paramterization.
  virtual void parseValues()
  {
    //std::cout << "PARSING VINEYARD PARAMETERS" << std::endl;
    base_height = 0.0;

    std::vector<float> xVertex, yVertex;
    parsePrimitive<float>(true, H, "height");
    parsePrimitive<float>(false, base_height, "baseHeight");
    parsePrimitive<float>(true, understory_height, "understoryHeight");
    parseMultiPrimitives<float>(false, xVertex, "xVertex");
    parseMultiPrimitives<float>(false, yVertex, "yVertex");

    int nNodes = xVertex.size();
    polygonVertices.resize(nNodes + 1);
    for (int k = 0; k < nNodes; k++) {
      polygonVertices[k].x_poly = xVertex[k];
      polygonVertices[k].y_poly = yVertex[k];
    }
    polygonVertices[nNodes].x_poly = xVertex[0];
    polygonVertices[nNodes].y_poly = yVertex[0];

    parsePrimitive<bool>(true, thinFence, "thinFence");// 1 to use thin fence approximation
    parsePrimitive<float>(true, rowSpacing, "rowSpacing");
    parsePrimitive<float>(true, rowWidth, "rowWidth");
    parsePrimitive<float>(true, rowAngle, "rowAngle");

    parsePrimitive<float>(true, LAD_eff, "LAD_eff");//leaf area per volume of canopy, including aisles. Effective (bulk) LAD. Re-calculated by the code using beta if thinFence==1 (=0.6667 for 2013 validation cases)
    parsePrimitive<float>(true, LAD_avg, "LAD_avg");//vertically averaged LAD for the rowitself, used in UD zone drag calc for vegetative rows only. (=4.3138 for 2013 validation cases)
    parsePrimitive<float>(true, beta, "opticalPorosity");
    parsePrimitive<float>(true, tkeMax, "tkeMax");
    parsePrimitive<float>(false, stdw, "upstreamSigmaW");
    parsePrimitive<float>(false, uustar, "upstreamUstar");//upstream ustar
    parsePrimitive<float>(false, d_v, "displacementHeightParallel");//displacement height in row-parallel winds

    //x_start += UID->simParams->halo_x;
    //y_start += UID->simParams->halo_y;
    double pi = 3.14159265359;
    canopy_rotation *= pi / 180.0;
    polygonVertices.resize(5);
  }

  void setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id);
  void orthog_vec(float, float[2], float[2], float[2], float[2]);
  void canopyVegetation(WINDSGeneralData *wgd, int building_id);
  void canopyWake(WINDSGeneralData *wgd, int building_id);
  void canopyTurbulenceWake(WINDSGeneralData *, TURBGeneralData *, int);
  void canopyStress(WINDSGeneralData *, TURBGeneralData *, int);
  int getCellFlagCanopy();
  int getCellFlagWake();
  float P2L(float[2], float[2], float[2]);

private:
  //float attenuationCoeff=1.0;
  float upwind_dir_unit = 0.;
  float understory_height;
  std::vector<float> tkeFac;
  std::vector<float> vineLm;
  float sinA, cosA;// sin cos of row angle
  float beta;// optical porosity
  float a_obf;// bleed flow aerodynamic porosity
  float stdw;// upstream vertical variance
  float uustar;// upstream ustar
  float d_v;// displacement height for whole ROC (depends on wind angle)
  float rowSpacing;
  float rowWidth;
  float rowAngle;
  bool thinFence = 0;
  float LAD_eff;
  float LAD_avg;
  std::map<int, float> u0, v0;
  float rL, z0_site;
  float tkeMax;

  // Geometry params (shared by veg and wake portions)
  float Rx_o[2];
  float Ry_o[2];
  float betaAngle;
  float d_dw;
  float szo_slope, szo_top, szo_bot, szo_top_uw;
  float z_mid, k_mid;// mid-canopy height and corresponding k-node
  float z_b;// base of the canopy
  float z_rel;// height off the terrain

  // Wake parameterization params (shared by veg and wake portions)
  float spreadrate_top, spreadrate_bot;
  float szt_uw, szt_local, szt_Lm;// shear zone thickness variables

  float a_local = 1;// the attenuation due to the closest upwind row only
  float tkeFacU_local;
  float tkeFacU = 0;
  float tkeFacV = 0;


  float u_c0, v_c0;// u0 and v0 rotated into row-aligned coordinates
  float u_c, v_c;// altered (parameterized) u and v, in row-aligned coordinates
  int N_e;// number of rows in entry region

  // V_c parameterization params (shared by veg and wake portions)
  float ustar_v_c,
    psi, psiH, dpsiH, ex, exH, dexH, vH_c, a_v, vref_c, zref, exRef, dexRef, psiRef, dpsiRef;
  int icell_face_ref, zref_k;
};

inline int CanopyROC::getCellFlagCanopy()
{
  return 26;
}

inline int CanopyROC::getCellFlagWake()
{
  return 27;
}

inline float CanopyROC::P2L(float P[2], float Lx[2], float Ly[2])
{
  return abs((Lx[1] - Lx[0]) * (Ly[0] - P[1]) - (Lx[0] - P[0]) * (Ly[1] - Ly[0])) / sqrt(pow(Lx[1] - Lx[0], 2) + pow(Ly[1] - Ly[0], 2));
}

