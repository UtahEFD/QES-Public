/*
 * QES-Winds
 *
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
 *
 */


#pragma once

/*
 * This class is an abstract representation of a building. It holds
 * the basic information and functions that every building should have.
*/
#include "util/ParseInterface.h"
#include "PolygonVertex.h"
#include "CutVertex.h"

using namespace std;

// forward declaration of WINDSInputData and WINDSGeneralData, which
// will be used by the derived classes and thus included there in the
// C++ files
class WINDSInputData;
class WINDSGeneralData;
class TURBGeneralData;

class Building : public ParseInterface
{
protected:

public:

	float building_rotation;
	float canopy_rotation;
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
	int i_building_cent, j_building_cent;


	double u0_h, v0_h; 				/**< u/v velocity at the height of building at the centroid */

	float width_eff;					/**< Effective width of the building */
	float length_eff;					/**< Effective length of the building */
	float small_dimension;		/**< Smaller of the height (H) and the effective cross-wind width (Weff) */
	float long_dimension;			/**< Larger of the height (H) and the effective cross-wind width (Weff) */

	float L_over_H, W_over_H;			/**< Length/width over height of the building */
	float Lr;										/**< Length of far wake zone */

	std::vector <polyVert> polygonVertices;

    Building()
    {
    }

    virtual ~Building()
    {
    }


    virtual void parseValues() = 0;

		virtual void setPolyBuilding (WINDSGeneralData* WGD)
		{
		}

    // Need to finalize the parameters here...
    virtual void setCellFlags (const WINDSInputData* WID, WINDSGeneralData* WGD, int building_number)
    {
    }

		virtual void upwindCavity (const WINDSInputData* WID, WINDSGeneralData* WGD)
    {
    }


    virtual void polygonWake (const WINDSInputData* WID, WINDSGeneralData* WGD, int building_id)
    {
    }

    virtual void canopyVegetation (WINDSGeneralData *WGD)
    {
    }

		virtual void streetCanyon (WINDSGeneralData *WGD)
		{
		}

		virtual void sideWall (const WINDSInputData* WID, WINDSGeneralData* WGD)
		{
		}

		virtual void rooftop (const WINDSInputData* WID, WINDSGeneralData* WGD)
		{
		}

    virtual void NonLocalMixing (WINDSGeneralData* WGD, TURBGeneralData* TGD,int buidling_id)
    {
    }
};
