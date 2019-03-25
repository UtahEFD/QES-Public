#pragma once

/*
 * This class represents a building that is a block with a length width height
 * and origin position. Rectangular buildigns may also have a rotation.
 */

#include "util/ParseInterface.h"
#include "NonPolyBuilding.h"

#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;


class RectangularBuilding : public NonPolyBuilding
{
private:

	int icell_cent, icell_cut;


public:

	RectangularBuilding()
	{
		buildingGeometry = 1;

	}

	RectangularBuilding(float xStart, float yStart, float bh, float length, float width, float height, std::vector<float> z)
	{
		buildingGeometry = 1;

		groupID = 999;
		buildingType = 5;
		H = height;
		baseHeight = bh;
		x_start = xStart;
		y_start = yStart;
		L = length;
		W = width;


	}

	virtual void parseValues()
	{
		parsePrimitive<int>(true, groupID, "groupID");
		parsePrimitive<int>(true, buildingType, "buildingType");
		parsePrimitive<float>(true, H, "height");
		parsePrimitive<float>(true, baseHeight, "baseHeight");
		parsePrimitive<float>(true, x_start, "xStart");
		parsePrimitive<float>(true, y_start, "yStart");
		parsePrimitive<float>(true, L, "length");
		parsePrimitive<float>(true, W, "width");

	}


	void setCutCells(float dx, float dy, std::vector<float> dz_array, std::vector<float> z, int nx, int ny, int nz, std::vector<int> &icellflag,
					 std::vector<std::vector<std::vector<float>>> &x_cut,std::vector<std::vector<std::vector<float>>> &y_cut,
					 std::vector<std::vector<std::vector<float>>> &z_cut, std::vector<std::vector<int>> &num_points,
					 std::vector<std::vector<float>> &coeff)
	{


		/// intersection points hard-coded
		for (int j=j_cut_start+1; j<j_cut_end; j++){
			for (int k=1; k<k_cut_end; k++){
				icell_cut = i_cut_start + j*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][2] = num_points[icell_cut][3] =
					num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][1] = 0;

					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k];

					x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] =
																									x_start-i_cut_start*dx;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz_array[k];

					x_cut[icell_cut][4][0] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] =
																									x_start-i_cut_start*dx;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;

				}
				icell_cut = (i_cut_end-1) + j*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] =
					num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][0] = 0;

					y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
					y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
					z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
					z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

					x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = dx;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][0] = x_cut[icell_cut][3][1] = x_cut[icell_cut][3][0] =
																										x_start+L-i_end*dx;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][1] = z_cut[icell_cut][3][2] = z_cut[icell_cut][3][1] = 0.0;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz_array[k];

					x_cut[icell_cut][4][2] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = dx;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][0] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][0] =
																										x_start+L-i_end*dx;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][2] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][0] = y_cut[icell_cut][5][3] = dy;
				}
			}
		}

		for (int i=i_cut_start+1; i<i_cut_end; i++){
			for (int k=1; k<k_cut_end; k++){
				icell_cut = i + j_cut_start*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] =
					num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][3] = 0;

					x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz_array[k];

					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] =
																									y_start-j_cut_start*dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

					x_cut[icell_cut][4][0] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] =
																									y_start-j_cut_start*dy;
				}
				icell_cut = i + (j_cut_end-1)*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][3] =
					num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][2] = 0;

					x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
					x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
					z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
					z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz_array[k];

					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] =
																										y_start+W-j_end*dy;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

					x_cut[icell_cut][4][0] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][0] = 0.0;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = dx;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][1] = y_cut[icell_cut][5][2] =
																										y_start+W-j_end*dy;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][3] = y_cut[icell_cut][5][0] = dy;
				}
			}
		}

		for (int i=i_cut_start+1; i<i_cut_end; i++){
			for (int j=j_cut_start+1; j<j_cut_end; j++){
				icell_cut = i + j*(nx-1) + k_end*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] =
					num_points[icell_cut][3] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][4] = 0;

					x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
					x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
					y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;

					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] =
																											H-(z[k_end-1]+0.5*dz_array[k_end]);
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k_end];

					x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] =
																											H-(z[k_end-1]+0.5*dz_array[k_end]);
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz_array[k_end];
				}
			}
		}

		for (int k=1; k<k_end; k++){
			icell_cut = i_cut_start + j_cut_start*(nx-1) + k*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){
				num_points[icell_cut][0] = num_points[icell_cut][1] =num_points[icell_cut][2] =num_points[icell_cut][3] = 4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k];

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = y_start-j_cut_start*dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz_array[k];

				x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = x_start-i_cut_start*dx;
				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz_array[k];
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][1] = 0.0;

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][1] = 0.0;
				x_cut[icell_cut][4][2] = x_cut[icell_cut][4][3] = dx;
				x_cut[icell_cut][4][4] = x_cut[icell_cut][4][5] = x_start-i_cut_start*dx;
				y_cut[icell_cut][4][1] = y_cut[icell_cut][4][2] = 0.0;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][5] = dy;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][3] = y_start-j_cut_start*dy;

				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][1] = 0.0;
				x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = dx;
				x_cut[icell_cut][5][4] = x_cut[icell_cut][5][5] = x_start-i_cut_start*dx;
				y_cut[icell_cut][5][1] = y_cut[icell_cut][5][2] = 0.0;
				y_cut[icell_cut][5][3] = y_cut[icell_cut][5][4] = y_start-j_cut_start*dy;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][5] = dy;

			}
		}

		for (int k=1; k<k_end; k++){
			icell_cut = (i_cut_end-1) + j_cut_start*(nx-1) + k*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){

				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3]=4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][0] = y_cut[icell_cut][0][1] = 0.0;
				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_start-j_cut_start*dy;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k];
				z_cut[icell_cut][0][1] = z_cut[icell_cut][0][2] = 0.0;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][1] = 0.0;
				x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = dx;
				z_cut[icell_cut][2][1] = z_cut[icell_cut][2][2] = 0.0;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = dz_array[k];

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][1] = x_start+L-i_end*dx;
				x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = dx;
				z_cut[icell_cut][3][1] = z_cut[icell_cut][3][2] = 0.0;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz_array[k];

				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][3] = dy;
				y_cut[icell_cut][4][1] = y_cut[icell_cut][4][2] = 0.0;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][5] = y_start-j_cut_start*dy;
				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][1] = 0.0;
				x_cut[icell_cut][4][2] = x_cut[icell_cut][4][3] = dx;
				x_cut[icell_cut][4][4] = x_cut[icell_cut][4][5] = x_start+L-i_end*dx;

				y_cut[icell_cut][5][4] = y_cut[icell_cut][5][3] = dy;
				y_cut[icell_cut][5][1] = y_cut[icell_cut][5][2] = 0.0;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][5] = y_start-j_cut_start*dy;
				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][1] = 0.0;
				x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = dx;
				x_cut[icell_cut][5][4] = x_cut[icell_cut][5][5] = x_start+L-i_end*dx;

			}
		}

		for (int k=1; k<k_end; k++){
			icell_cut = i_cut_start + (j_cut_end-1)*(nx-1) + k*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){


				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3]=4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k];

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = y_start+W-j_end*dy;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

				x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = x_start-i_cut_start*dx;
				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = dz_array[k];
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][1] = 0.0;

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz_array[k];

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][1] = 0.0;
				x_cut[icell_cut][4][2] = x_cut[icell_cut][4][3] = x_start-i_cut_start*dx;
				x_cut[icell_cut][4][4] = x_cut[icell_cut][4][5] = dx;
				y_cut[icell_cut][4][1] = y_cut[icell_cut][4][2] = 0.0;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][5] = dy;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][3] = y_start+W-j_end*dy;

				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][1] = 0.0;
				x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = x_start-i_cut_start*dx;
				x_cut[icell_cut][5][4] = x_cut[icell_cut][5][5] = dx;
				y_cut[icell_cut][5][1] = y_cut[icell_cut][5][2] = 0.0;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][5] = dy;
				y_cut[icell_cut][5][4] = y_cut[icell_cut][5][3] = y_start+W-j_end*dy;

			}
		}

		for (int k=1; k<k_end; k++){
			icell_cut = (i_cut_end-1) + (j_cut_end-1)*(nx-1) + k*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){


				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3]=4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][0] = y_cut[icell_cut][0][1] = y_start+W-j_end*dy;
				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k];
				z_cut[icell_cut][0][1] = z_cut[icell_cut][0][2] = 0.0;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][1] = x_start+L-i_end*dx;
				x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = dx;
				z_cut[icell_cut][2][1] = z_cut[icell_cut][2][2] = 0.0;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = dz_array[k];

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][1] = 0.0;
				x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = dx;
				z_cut[icell_cut][3][1] = z_cut[icell_cut][3][2] = 0.0;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz_array[k];

				y_cut[icell_cut][4][3] = y_cut[icell_cut][4][4] = 0.0;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][5] = dy;
				y_cut[icell_cut][4][1] = y_cut[icell_cut][4][2] = y_start+W-j_end*dy;
				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][1] = 0.0;
				x_cut[icell_cut][4][2] = x_cut[icell_cut][4][3] = x_start+L-i_end*dx;
				x_cut[icell_cut][4][4] = x_cut[icell_cut][4][5] = dx;

				y_cut[icell_cut][5][3] = y_cut[icell_cut][5][4] = 0.0;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][5] = dy;
				y_cut[icell_cut][5][1] = y_cut[icell_cut][5][2] = y_start+W-j_end*dy;
				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][1] = 0.0;
				x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = x_start+L-i_end*dx;
				x_cut[icell_cut][5][4] = x_cut[icell_cut][5][5] = dx;
			}
		}

			icell_cut = i_cut_start + j_cut_start*(nx-1) + k_end*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){
				num_points[icell_cut][0] = num_points[icell_cut][2] = num_points[icell_cut][5] = 4;
				num_points[icell_cut][1] = num_points[icell_cut][3] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k_end];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz_array[k_end];

				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;

				y_cut[icell_cut][1][4] = y_cut[icell_cut][1][5] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				y_cut[icell_cut][1][3] = y_cut[icell_cut][1][2] = y_start-j_cut_start*dy;
				z_cut[icell_cut][1][4] = z_cut[icell_cut][1][3] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][5] = dz_array[k_end];

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][5] = 0.0;
				x_cut[icell_cut][3][3] = x_cut[icell_cut][3][4] = dx;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = x_start-i_cut_start*dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][3][4] = z_cut[icell_cut][3][5] = dz_array[k_end];

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][5] = 0.0;
				x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = dx;
				x_cut[icell_cut][4][3] = x_cut[icell_cut][4][4] = x_start-i_cut_start*dx;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = 0.0;
				y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_start-j_cut_start*dy;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][5] = dy;
			}

			icell_cut = (i_cut_end-1) + j_cut_start*(nx-1) + k_end*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){
				num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][5] = 4;
				num_points[icell_cut][0] = num_points[icell_cut][3] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k_end];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz_array[k_end];

				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;

				y_cut[icell_cut][0][4] = y_cut[icell_cut][0][5] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				y_cut[icell_cut][0][3] = y_cut[icell_cut][0][2] = y_start-j_cut_start*dy;
				z_cut[icell_cut][0][4] = z_cut[icell_cut][0][3] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][5] = dz_array[k_end];

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][5] = 0.0;
				x_cut[icell_cut][3][3] = x_cut[icell_cut][3][4] = dx;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = x_start+L-i_end*dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = 0.0;
				z_cut[icell_cut][3][4] = z_cut[icell_cut][3][5] = dz_array[k_end];

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][5] = 0.0;
				x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = dx;
				x_cut[icell_cut][4][3] = x_cut[icell_cut][4][4] = x_start+L-i_end*dx;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = 0.0;
				y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = dy;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][5] = y_start-j_cut_start*dy;
			}

			icell_cut = i_cut_start + (j_cut_end-1)*(nx-1) + k_end*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){
				num_points[icell_cut][0] = num_points[icell_cut][3] = num_points[icell_cut][5] = 4;
				num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz_array[k_end];

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz_array[k_end];

				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;

				y_cut[icell_cut][1][4] = y_cut[icell_cut][1][5] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				y_cut[icell_cut][1][3] = y_cut[icell_cut][1][2] = y_start+W-j_end*dy;
				z_cut[icell_cut][1][4] = z_cut[icell_cut][1][3] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][5] = dz_array[k_end];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][5] = 0.0;
				x_cut[icell_cut][2][3] = x_cut[icell_cut][2][4] = dx;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_start-i_cut_start*dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][2][4] = z_cut[icell_cut][2][5] = dz_array[k_end];

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][5] = 0.0;
				x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_start-i_cut_start*dx;
				x_cut[icell_cut][4][3] = x_cut[icell_cut][4][4] = dx;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = 0.0;
				y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_start+W-j_end*dy;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][5] = dy;
			}

			icell_cut = (i_cut_end-1) + (j_cut_end-1)*(nx-1) + k_end*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){
				num_points[icell_cut][1] = num_points[icell_cut][3] = num_points[icell_cut][5] = 4;
				num_points[icell_cut][0] = num_points[icell_cut][2] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz_array[k_end];

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz_array[k_end];

				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;

				y_cut[icell_cut][0][4] = y_cut[icell_cut][0][5] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				y_cut[icell_cut][0][3] = y_cut[icell_cut][0][2] = y_start+W-j_end*dy;
				z_cut[icell_cut][0][4] = z_cut[icell_cut][0][3] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][5] = dz_array[k_end];

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][5] = 0.0;
				x_cut[icell_cut][2][3] = x_cut[icell_cut][2][4] = dx;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_start+L-i_end*dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = H-(z[k_end-1]+0.5*dz_array[k_end]);
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = 0.0;
				z_cut[icell_cut][2][4] = z_cut[icell_cut][2][5] = dz_array[k_end];

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][5] = 0.0;
				x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_start+L-i_end*dx;
				x_cut[icell_cut][4][3] = x_cut[icell_cut][4][4] = dx;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_start+W-j_end*dy;
				y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = 0.0;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][5] = dy;
			}



	}


	void setCellsFlag(float dx, float dy, std::vector<float> dz_array, int nx, int ny, int nz, std::vector<float> z, std::vector<int> &icellflag, int mesh_type_flag)
	{

		if (mesh_type_flag == 1)
		{
			// defining building and cut-cell indices
			if (fmod(x_start,dx)==0)
			{
				i_start = x_start/dx;
				i_cut_start = i_start;
			}
			else
			{
				i_start = 1+x_start/dx;
				i_cut_start = x_start/dx;
			}
			i_end = (x_start+L)/dx;
			i_cut_end = i_end;
			if (fmod((x_start+L),dx)!=0)
			{
				i_cut_end = 1+(x_start+L)/dx;
			}
			if (fmod(y_start,dy)==0)
			{
				j_start = y_start/dy;
				j_cut_start = j_start;
			}
			else
			{
				j_start = 1+y_start/dy;
				j_cut_start = y_start/dy;
			}
			j_end = (y_start+W)/dy;
			j_cut_end = j_end;
			if (fmod((y_start+W),dy)!=0)
			{
				j_cut_end = 1+(y_start+W)/dy;
			}
			for (auto k=1; k<z.size(); k++)
			{
				k_start = k;
				if (baseHeight <= z[k])
				{
					break;
				}
			}

			for (auto k=k_start; k<z.size(); k++)
			{
				k_end = k+1;
				k_cut_end = k_end+1;
				if (baseHeight+H < z[k+1])
				{
					if (baseHeight+H >z[k]+0.5*dz_array[k+1])
					{
						k_end = k+1;
						k_cut_end = k_end+1;
					}
					if (baseHeight+H < z[k]+0.5*dz_array[k+1])
					{
						k_end = k;
						k_cut_end = k_end+1;
					}
					if (baseHeight+H == z[k]+0.5*dz_array[k+1])
					{
						k_end = k_cut_end = k+1;
					}
					break;
				}
			}


#if 0
    		std::cout << "i_start:" << i_start << "\n";
   		 	std::cout << "i_end:" << i_end << "\n";
    		std::cout << "j_start:" << j_start << "\n";
   		 	std::cout << "j_end:" << j_end << "\n";
   		 	std::cout << "k_end:" << k_end << "\n";
				std::cout << "k_start:" << k_start << "\n";

   		 	std::cout << "i_cut_start:" << i_cut_start << "\n";
    		std::cout << "i_cut_end:" << i_cut_end << "\n";
    		std::cout << "j_cut_start:" << j_cut_start << "\n";
    		std::cout << "j_cut_end:" << j_cut_end << "\n";
    		std::cout << "k_cut_end:" << k_cut_end << "\n";

#endif

			for (int k = 1; k < k_cut_end; k++){
				for (int j = j_cut_start; j < j_cut_end; j++){
					for (int i = i_cut_start; i < i_cut_end; i++){
		               	icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
						icellflag[icell_cent] = 7;                         /// Set cell index flag to cut-cell
					}
				}
   			}


   			for (int k = k_start; k < k_end; k++){
    	   		for (int j = j_start; j < j_end; j++){
    	       		for (int i = i_start; i < i_end; i++){
    	            	icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
						icellflag[icell_cent] = 0;                         /// Set cell index flag to building
					}
				}
    		}

		}
		else
		{
			i_start = std::round(x_start/dx);     // Index of building start location in x-direction
      i_end = std::round((x_start+L)/dx);   // Index of building end location in x-direction
      j_start = std::round(y_start/dy);     // Index of building start location in y-direction
      j_end = std::round((y_start+W)/dy);   // Index of building end location in y-direction

			for (auto k=1; k<z.size(); k++)
			{
				k_start = k;
				if (baseHeight <= z[k])
				{
					break;
				}
			}

			for (auto k=k_start; k<z.size(); k++)
			{
				k_end = k+1;
				if (baseHeight+H < z[k+1])
				{
					break;
				}
			}

#if 0
    		std::cout << "i_start:" << i_start << "\n";
    		std::cout << "i_end:" << i_end << "\n";
    		std::cout << "j_start:" << j_start << "\n";
    		std::cout << "j_end:" << j_end << "\n";
    		std::cout << "k_end:" << k_end << "\n";
#endif


   			for (int k = k_start; k < k_end; k++)
				{
					for (int j = j_start; j < j_end; j++)
					{
						for (int i = i_start; i < i_end; i++)
						{
							icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
							icellflag[icell_cent] = 0;                         /// Set cell index flag to building
						}
					}
				}





		}
	}



};
