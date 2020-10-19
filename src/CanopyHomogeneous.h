#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "Canopy.h"

class CanopyHomogeneous : public Canopy
{
public:
    
    CanopyHomogeneous()
    {
    }
    
    
    virtual void parseValues()
    {
        parsePrimitive<float>(true, attenuationCoeff, "attenuationCoefficient");
        parsePrimitive<float>(true, H, "height");
        parsePrimitive<float>(true, base_height, "baseHeight");
        parsePrimitive<float>(true, x_start, "xStart");
        parsePrimitive<float>(true, y_start, "yStart");
        parsePrimitive<float>(true, L, "length");
        parsePrimitive<float>(true, W, "width");
        parsePrimitive<float>(true, canopy_rotation, "canopyRotation");
    
    
        //x_start += UID->simParams->halo_x;
        //y_start += UID->simParams->halo_y;
        canopy_rotation *= M_PI/180.0;
        polygonVertices.resize (5);
        polygonVertices[0].x_poly = polygonVertices[4].x_poly = x_start;
        polygonVertices[0].y_poly = polygonVertices[4].y_poly = y_start;
        polygonVertices[1].x_poly = x_start-W*sin(canopy_rotation);
        polygonVertices[1].y_poly = y_start+W*cos(canopy_rotation);
        polygonVertices[2].x_poly = polygonVertices[1].x_poly+L*cos(canopy_rotation);
        polygonVertices[2].y_poly = polygonVertices[1].y_poly+L*sin(canopy_rotation);
        polygonVertices[3].x_poly = x_start+L*cos(canopy_rotation);
        polygonVertices[3].y_poly = y_start+L*sin(canopy_rotation);

    }
    

    void canopyInitial(WINDSGeneralData *wgd, int building_id);

    void canopyVegetation(WINDSGeneralData *wgd, int building_id);
  
    /*!
     * This function takes in variables read in from input files and initializes required variables for definig
     * canopy elementa.
     */
    //void readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies, int &lu_canopy_flag,
    //	std::vector<std::vector<std::vector<float>>> &canopy_atten,std::vector<std::vector<float>> &canopy_top);
  
private:
  
    float attenuationCoeff;
    const int cellFlagCionco=11;
  
};
