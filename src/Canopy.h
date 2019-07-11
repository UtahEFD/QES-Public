#pragma once

#include "util/ParseInterface.h"
#include "Building.h"
#include <cmath>


class Canopy : public Building
{
private:

public:

	float atten;
	int landuse_veg_flag, landuse_urb_flag, lu_canopy_flag;
	int canopy_flag;

	virtual void parseValues()
	{

		parsePrimitive<float>(true, atten, "attenuationCoefficient");
		parsePrimitive<float>(true, H, "height");
		parsePrimitive<float>(true, baseHeight, "baseHeight");
		parsePrimitive<float>(true, x_start, "xStart");
		parsePrimitive<float>(true, y_start, "yStart");
		parsePrimitive<float>(true, L, "length");
		parsePrimitive<float>(true, W, "width");



	}


	/*!
	 *This function takes in variables read in from input files and initializes required variables for definig
	 *canopy elementa.
	 */
	void readCanopy(int nx, int ny, int nz, int landuse_flag, int num_canopies, int &lu_canopy_flag,
					std::vector<std::vector<std::vector<float>>> &canopy_atten,std::vector<std::vector<float>> &canopy_top);

	/*!
	 *This function takes in icellflag defined in the defineCanopy function along with variables initialized in
	 *the readCanopy function and initial velocity field components (u0 and v0). This function applies the urban canopy
	 *parameterization and returns modified initial velocity field components.
	 */

	void plantInitial(int nx, int ny, int nz, float vk, std::vector<int> &icellflag, std::vector<float> z, std::vector<double> &u0,
						std::vector<double> &v0, std::vector<std::vector<std::vector<float>>> &canopy_atten,
						std::vector<std::vector<float>> &canopy_top, std::vector<std::vector<float>> &canopy_top_index,
						std::vector<std::vector<float>> &canopy_ustar, std::vector<std::vector<float>> &canopy_z0,
						std::vector<std::vector<float>> &canopy_d);

	/*!
	 *This function is being call from the plantInitial function and uses linear regression method to define ustar and
	 *surface roughness of the canopy.
	 */

	void regression(int nx, int ny, int nz, float vk, float *z, double *u0, double *v0,
					std::vector<std::vector<std::vector<float>>> &canopy_atten, std::vector<std::vector<float>> &canopy_top,
					std::vector<std::vector<float>> &canopy_top_index, std::vector<std::vector<float>> &canopy_ustar,
					std::vector<std::vector<float>> &canopy_z0);

	/*!
	 *This function is being called from the plantInitial function and uses the bisection method to find the displacement
	 *height of the canopy.
	 */

	float bisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

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

	void defineCanopy(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag, int num_canopies,
						int lu_canopy_flag, std::vector<std::vector<std::vector<float>>> &canopy_atten,
						std::vector<std::vector<float>> &canopy_top)
	{

		i_start = std::round(x_start/dx);       // Index of canopy start location in x-direction
		i_end = std::round((x_start+L)/dx);			// Index of canopy end location in x-direction
		j_end = std::round((y_start+W)/dy);			// Index of canopy end location in y-direction
		j_start = std::round(y_start/dy);				// Index of canopy start location in y-direction
		k_start = std::round(baseHeight/dz);			// Index of canopy start location in z-direction
		k_end = std::round((baseHeight+H)/dz)+1;				// Index of canopy end location in y-direction

#if 0
    	std::cout << "i_start:" << i_start << "\n";
   	 	std::cout << "i_end:" << i_end << "\n";
    	std::cout << "j_start:" << j_start << "\n";
   	 	std::cout << "j_end:" << j_end << "\n";
   	 	std::cout << "k_end:" << k_end << "\n";
#endif

		for (int j=j_start; j<j_end; j++)
		{
			for (int i=i_start; i<i_end; i++)
			{
				for (int k=k_start; k<k_end; k++)
				{
					int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
					if (icellflag[icell_cent] != 0)       // if the cell is not solid
					{
						if (lu_canopy_flag > 0)
						{
						}
						else
						{
							if( H > canopy_top[i][j])
							{
								canopy_top[i][j] = H;
							}
							canopy_atten[i][j][k] = atten;     // initiate all attenuation coefficients to the canopy coefficient
						}
					}
				}
			}
		}

	}


};
