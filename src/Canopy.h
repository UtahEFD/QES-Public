#pragma once

#include <cmath>
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
    
    CanopyType _cType;
    
protected: 
    /*!
     * This function takes in variables initialized by the readCanopy function and sets the boundaries of 
     * the canopy and defines initial values for the canopy height.
     */
    void canopyDefineBoundary(WINDSGeneralData *wgd,int cellFlagToUse);
    
      
    /*!
     * For there and below, the canopyInitial function has to be defined
     */
    virtual void canopyInitial(WINDSGeneralData *wgd) = 0;
    
    /*!
     * For there and below, the canopyVegetation function has to be defined
     */
    virtual void canopyVegetation(WINDSGeneralData *wgd) = 0;
    
    /*!
     * 
     */
    std::vector<float> canopy_atten;	  /**< Canopy attenuation coefficient */
    std::vector<float> canopy_bot;		  /**< Canopy bottom */
    std::vector<float> canopy_top;		  /**< Canopy top */
    
    std::vector<int> canopy_bot_index;	  /**< Canopy bottom index */
    std::vector<int> canopy_top_index;	  /**< Canopy top index */

    std::vector<float> canopy_base;	      /**< Canopy base */
    std::vector<float> canopy_height;	  /**< Canopy height */

    std::vector<float> canopy_z0;		  /**< Canopy surface roughness */
    std::vector<float> canopy_ustar;	  /**< Velocity gradient at the top of canopy */
    std::vector<float> canopy_d;		  /**< Canopy displacement length */
    
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
