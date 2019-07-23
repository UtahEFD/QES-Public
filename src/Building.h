#pragma once

/*
 * This class is an abstract representation of a building. It holds
 * the basic information and functions that every building should have.
*/

#include "util/ParseInterface.h"

class Building : public ParseInterface
{
protected:

public:

	float building_rotation;
	float x_start;
	float y_start;
	float L;									/**< Length of the building */
	float W;                  /**< Width of the building */
	int i_start, i_end, j_start, j_end, k_end,k_start;				// Indices of start and end of building
	 																													// in stair-step method
	int i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;  // Indices of start and end of building
	 																																// in cut-cell method
	float H;							/**< Height of the building */
	float base_height; 			/**< Base height of the building */

	float upwind_dir;						/**< Wind direction of initial velocity at the height of building at the centroid */
	float height_eff;						/**< Effective height of the building */
	float building_cent_x, building_cent_y;				/**< Coordinates of centroid of the building */

	double u0_h, v0_h; 				/**< u/v velocity at the height of building at the centroid */

	float width_eff;					/**< Effective width of the building */
	float length_eff;					/**< Effective length of the building */

	float L_over_H, W_over_H;			/**< Length/width over height of the building */
	float Lr;										/**< Length of far wake zone */

	virtual void parseValues() = 0;

    // virtual functions....

    // Need to finalize the parameters here...
    virtual void setCellFlags()
    {
    }

    virtual void setCellFlags(const URBInputData *UID, URBGeneralData *ugd)
    {
		}

    virtual void polygonWake(const URBInputData *UID, URBGeneralData *ugd)
    {
    }

    virtual void callParameterizationTwo()
    {
    }

    virtual void callParameterizationSpecial()
    {
    }

};
