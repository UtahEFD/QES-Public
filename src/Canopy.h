#pragma once

#include <cmath>

#include "util/ParseInterface.h"

#include "PolyBuilding.h"

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

	float atten;

	Canopy()
	{

	}


	virtual void parseValues()
	{
		parsePrimitive<float>(true, atten, "attenuationCoefficient");
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

  void canopyVegetation(URBGeneralData *ugd);


	/*!
	 *This function takes in variables read in from input files and initializes required variables for definig
	 *canopy elementa.
	 */
	//void readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies, int &lu_canopy_flag,
				//	std::vector<std::vector<std::vector<float>>> &canopy_atten,std::vector<std::vector<float>> &canopy_top);

	/*!
	 *This function takes in icellflag defined in the defineCanopy function along with variables initialized in
	 *the readCanopy function and initial velocity field components (u0 and v0). This function applies the urban canopy
	 *parameterization and returns modified initial velocity field components.
	 */

	void plantInitial(URBGeneralData *ugd);

	/*!
	 *This function is being call from the plantInitial function and uses linear regression method to define ustar and
	 *surface roughness of the canopy.
	 */

	void regression(URBGeneralData *ugd);


	/*!
	 *This is a new function wrote by Lucas Ulmer and is being called from the plantInitial function. The purpose of this
	 *function is to use bisection method to find root of the specified equation. It calculates the displacement height
	 *when the bisection function is not finding it.
	 */

	float canopy_slope_match(float z0, float canopy_top, float canopy_atten);

	/*!
	 *This function takes in variables initialized by the readCanopy function and sets the boundaries of the canopy and
	 *defines initial values for the canopy height and attenuation.
	 */

	 void defineCanopy(URBGeneralData *ugd);

};
