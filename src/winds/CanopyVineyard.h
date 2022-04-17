#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "Canopy.h"

class CanopyVineyard : public CanopyElement
{
public:
  CanopyVineyard()
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
    //parsePrimitive<float>(true, x_start, "xStart");
    //parsePrimitive<float>(true, y_start, "yStart");
   /* 
    parsePrimitive<float>(false, corner1x, "corner1x");
    parsePrimitive<float>(false, corner1y, "corner1y");
    parsePrimitive<float>(false, corner2x, "corner2x");
    parsePrimitive<float>(false, corner2y, "corner2y");
    parsePrimitive<float>(false, corner3x, "corner3x");
    parsePrimitive<float>(false, corner3y, "corner3y");
    parsePrimitive<float>(false, corner4x, "corner4x");
    parsePrimitive<float>(false, corner4y, "corner4y");
   */ 
    parseMultiPrimitives<float>(false,xVertex, "xVertex");
    parseMultiPrimitives<float>(false,yVertex, "yVertex");

    int nNodes = xVertex.size();
    polygonVertices.resize(nNodes+1);
    for (int k=0; k<nNodes; k++){
      polygonVertices[k].x_poly = xVertex[k];
      polygonVertices[k].y_poly = yVertex[k];
    }
    polygonVertices[nNodes].x_poly = xVertex[0];
    polygonVertices[nNodes].y_poly = yVertex[0];

    //for (int outi=0; outi<nNodes+1; outi++){
    //  std::cout << "polygonVertices[" << outi << "].x_poly = " << polygonVertices[outi].x_poly << ", polygonVertices[" << outi << "].y_poly = " << polygonVertices[outi].y_poly << std::endl;
    //}

    parsePrimitive<bool>(true, thinFence, "thinFence");// 1 to use thin fence approximation
    parsePrimitive<float>(true, rowSpacing, "rowSpacing");
    parsePrimitive<float>(true, rowWidth, "rowWidth");
    parsePrimitive<float>(true, rowAngle, "rowAngle");

    parsePrimitive<float>(false, LAD_eff, "LAD_eff");
    parsePrimitive<float>(false, LAD_avg, "LAD_avg");
    parsePrimitive<float>(true, beta, "opticalPorosity");
    parsePrimitive<float>(false, tkeMax, "tkeMax");
    parsePrimitive<float>(false, stdw, "upstreamSigmaW");
    parsePrimitive<float>(false, uustar, "upstreamUstar");//upstream ustar
    parsePrimitive<float>(false, d_v, "displacementHeightParallel");//upstream ustar
    //parsePrimitive<int>(false, wbModel, "fenceModel");
    //parsePrimitive<float>(false, fetch, "fetch");

    //x_start += UID->simParams->halo_x;
    //y_start += UID->simParams->halo_y;
    canopy_rotation *= M_PI / 180.0;
    polygonVertices.resize(5);
    /*        
	polygonVertices[0].x_poly = polygonVertices[4].x_poly = x_start;
        polygonVertices[0].y_poly = polygonVertices[4].y_poly = y_start;
        polygonVertices[1].x_poly = x_start-L*sin(canopy_rotation);
        polygonVertices[1].y_poly = y_start+L*cos(canopy_rotation);
        polygonVertices[2].x_poly = polygonVertices[1].x_poly+W*cos(canopy_rotation);
        polygonVertices[2].y_poly = polygonVertices[1].y_poly+W*sin(canopy_rotation);
        polygonVertices[3].x_poly = x_start+W*cos(canopy_rotation);
        polygonVertices[3].y_poly = y_start+W*sin(canopy_rotation);


    polygonVertices[0].x_poly = polygonVertices[4].x_poly = corner1x;
    polygonVertices[0].y_poly = polygonVertices[4].y_poly = corner1y;
    polygonVertices[1].x_poly = corner2x;
    polygonVertices[1].y_poly = corner2y;
    polygonVertices[2].x_poly = corner3x;
    polygonVertices[2].y_poly = corner3y;
    polygonVertices[3].x_poly = corner4x;
    polygonVertices[3].y_poly = corner4y;
*/  
  }

  void setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id);
  void orthog_vec(float, float[2], float[2], float[2], float[2]);
  void canopyVegetation(WINDSGeneralData *wgd, int building_id);
  void canopyWake(WINDSGeneralData *wgd, int building_id);
  void canopyTurbulenceWake(WINDSGeneralData *, TURBGeneralData *, int);
  int getCellFlagCanopy();
  int getCellFlagWake();
  float P2L(float[2], float[2], float[2]);


  /*!
     * This function takes in variables read in from input files and initializes required variables for definig
     * canopy elementa.
     */
  //void readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies, int &lu_canopy_flag,
  //	std::vector<std::vector<std::vector<float>>> &canopy_atten,std::vector<std::vector<float>> &canopy_top);

private:
  //float attenuationCoeff=1.0;
  float upwind_dir_unit = 0.;
  float understory_height;
  std::vector<float> tkeFac;
  std::vector<float> vineLm;
  float beta;// optical porosity
  //int wbModel = 2;   // flow in windbreak 1 for wang aerodynamic profile; bean otherwise
  float a_obf;// bleed flow areo porosity
  //float d = 0.0;	   // from upwind profile
  float stdw;// upstream vertical variance
  float uustar;// upstream ustar
  float d_v;// displacement height for whole vineyard (depends on wind angle)
  float rowSpacing;
  float rowWidth;
  float rowAngle;
  bool thinFence = 0;
  float LAD_eff=0.66667;//leaf area per volume of canopy, including aisles. Effective (bulk) LAD. Re-calculated by the code using beta if thinFence==1
  float LAD_avg=4.3138;//vertically averaged LAD for just the row, used in UD zone drag calc for vegetative rows only.  
  //float fetch = 7;
  //float corner1x, corner1y, corner2x, corner2y, corner3x, corner3y, corner4x, corner4y;
  std::map<int, float> u0, v0;
  float rL, z0_site;
  float tkeMax;
};

inline int CanopyVineyard::getCellFlagCanopy()
{
  return 26;
}

inline int CanopyVineyard::getCellFlagWake()
{
  return 27;
}

inline float CanopyVineyard::P2L(float P[2], float Lx[2], float Ly[2])
{
  return abs((Lx[1] - Lx[0]) * (Ly[0] - P[1]) - (Lx[0] - P[0]) * (Ly[1] - Ly[0])) / sqrt(pow(Lx[1] - Lx[0], 2) + pow(Ly[1] - Ly[0], 2));

}

