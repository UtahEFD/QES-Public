#pragma once

#include <cmath>
#include <map>
#include "util/ParseInterface.h"
#include "Building.h"

enum CanopyType {
    Homogeneous,
    IsolatedTree,
    Vineyard
};

class Canopy : public Building
{
private:
    
protected:
    
	float x_min, x_max, y_min, y_max;      // Minimum and maximum values of x
																				 // and y for a building
	float x_cent, y_cent;                  /**< Coordinates of center of a cell */
	float polygon_area;                    /**< Polygon area */
	//int icell_cent, icell_face;
    
    int nx_canopy, ny_canopy, nz_canopy;
    int numcell_cent_2d, numcell_cent_3d;

public:
    
    Canopy()
    { 
    }
    virtual ~Canopy()
    {
    }
    
    /*! 
     * For all Canopy classes derived, this need to be defined
     */
    virtual void parseValues() = 0;
    
    void setPolyBuilding (WINDSGeneralData* WGD);

    virtual void setCellFlags (const WINDSInputData* WID, WINDSGeneralData* WGD, int building_number) = 0;    
    virtual void canopyVegetation(WINDSGeneralData *wgd,int building_id) = 0;
    virtual void canopyWake(WINDSGeneralData *wgd,int building_id) = 0;

    virtual int getCellFlagCanopy() = 0;
    virtual int getCellFlagWake() = 0;

    std::map<int,int> canopy_cellMap2D,canopy_cellMap3D;    /**< map beteen WINDS grid and canopy grid */
    
    CanopyType _cType;
    
protected: 
          
    /*!
     * For there and below, the canopyInitial function has to be defined
     */
    virtual void setCanopyGrid(WINDSGeneralData *wgd,int building_id);
    
    
    /*!
     * 
     */
    std::vector<float> canopy_atten;	  /**< Canopy attenuation coefficient */
    
    std::vector<float> canopy_bot;		  /**< Canopy bottom */
    std::vector<int> canopy_bot_index;	  /**< Canopy bottom index */
    std::vector<float> canopy_top;		  /**< Canopy top */
    std::vector<int> canopy_top_index;	  /**< Canopy top index */

    std::vector<float> canopy_base;	      /**< Canopy base */
    std::vector<float> canopy_height;	  /**< Canopy height */

    std::vector<float> canopy_z0;		  /**< Canopy surface roughness */
    std::vector<float> canopy_ustar;	  /**< Velocity gradient at the top of canopy */
    std::vector<float> canopy_d;		  /**< Canopy displacement length */
    
    /*!
     * This function takes in icellflag defined in the defineCanopy function along with variables initialized in
     * the readCanopy function and initial velocity field components (u0 and v0). This function applies the urban 
     * canopy parameterization and returns modified initial velocity field components.
     */
    void canopyCioncoParam(WINDSGeneralData * wgd);
    
    /*!
     * This function is being call from the plantInitial function and uses linear regression method to define 
     * ustar and surface roughness of the canopy.
     */
    void canopyRegression(WINDSGeneralData *wgd);
    
    /*!
     * This is a new function wrote by Lucas Ulmer and is being called from the plantInitial function. The purpose 
     * of this function is to use bisection method to find root of the specified equation. It calculates the 
     * displacement height when the bisection function is not finding it.
     */
    float canopySlopeMatch(float z0, float canopy_top, float canopy_atten);
    
    /*!
     * 
     */
    float canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

private:
    

};
