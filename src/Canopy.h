#pragma once

#include <cmath>
#include "util/ParseInterface.h"
#include "Building.h"

enum CanopyType {
    Cionco,
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
	int icell_cent, icell_face;
    
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
    void canopyDefineBoundary(URBGeneralData *ugd,int cellFlagToUse);
    
    /*!
     * For there and below, the canopyVegetation function has to be defined
     */
    virtual void canopyVegetation(URBGeneralData *ugd) = 0;
    
private:
    
};
