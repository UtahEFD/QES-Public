#pragma once

#include "ParseInterface.h"
#include "NonPolyBuilding.h"
<<<<<<< HEAD
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

=======

#define CELL(i,j,k,sub) ((i) + (j) * ((nx) - (sub)) + (k) * ((nx) - (sub)) * ((ny) - (sub)))
>>>>>>> origin/doxygenAdd

class RectangularBuilding : public NonPolyBuilding
{
private:
<<<<<<< HEAD
	int i_start, i_end, j_start, j_end, k_end;
	int i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;
	int icell_cent, icell_cut;
	float H; 
=======

>>>>>>> origin/doxygenAdd

public:

	RectangularBuilding()
	{
		buildingGeometry = 1;
<<<<<<< HEAD

=======
		Lf = -999;
		Leff = 0;
		Weff = 0;
>>>>>>> origin/doxygenAdd
	}

	RectangularBuilding(float xfo, float yfo, float bh, float dx, float dy, float dz)
	{
		buildingGeometry = 1;
<<<<<<< HEAD

		groupID = 999;
		buildingType = 5;
		H = dz;
		baseHeight = bh;
		x_start = xfo;
		y_start = yfo;
		L = dx;
		W = dy;

=======
		Lf = -999;
		Leff = 0;
		Weff = 0;

		groupID = 999;
		buildingType = 5;
		height = dz;
		baseHeight = bh;
		centroidX = xfo + 0.5f * dx;
		centroidX = yfo + 0.5f * dy;
		xFo = xfo;
		yFo = yfo;
		length = dx;
		width = dy;
		rotation = 0.0f;
		
		Wt = 0.5 * width;
		Lt = 0.5 * length;
>>>>>>> origin/doxygenAdd
	}

	virtual void parseValues()
	{
		parsePrimitive<int>(true, groupID, "groupID");
		parsePrimitive<int>(true, buildingType, "buildingType");
<<<<<<< HEAD
		parsePrimitive<float>(true, H, "height");
		parsePrimitive<float>(true, baseHeight, "baseHeight");
		parsePrimitive<float>(true, x_start, "xFo");
		parsePrimitive<float>(true, y_start, "yFo");
		parsePrimitive<float>(true, L, "length");
		parsePrimitive<float>(true, W, "width");

	}

	void setBoundaries(float dx, float dy, float dz, int nx, int ny, int nz, float *zm, float *e, float *f, float *g, float *h, float *m, float *n, int *icellflag)
	{
		long numcell_cent = (nx-1)*(ny-1)*(nz-1);         /// Total number of cell-centered values in domain
    	long numface_cent = nx*ny*nz;                     /// Total number of face-centered values in domain
		
		/// defining building and cut-cell indices
		if (fmod(x_start,dx)==0){
			i_start = x_start/dx;
			i_cut_start = i_start;
		} else{
			i_start = 1+x_start/dx;
			i_cut_start = x_start/dx;
		}
		i_end = (x_start+L)/dx;
		i_cut_end = i_end;
		if (fmod((x_start+L),dx)!=0){
			i_cut_end = 1+(x_start+L)/dx;
		}
		if (fmod(y_start,dy)==0){	
			j_start = y_start/dy;
			j_cut_start = j_start;
		} else{
			j_start = 1+y_start/dy;
			j_cut_start = y_start/dy;
		}		
		j_end = (y_start+W)/dy;
		j_cut_end = j_end;
		if (fmod((y_start+W),dy)!=0){
			j_cut_end = 1+(y_start+W)/dy;
		}
		k_end = (H/dz)+1;
		k_cut_end = k_end;
		if (fmod(H,dz)!=0){
			k_cut_end = 2+(H/dz);
		}

		k_start = baseHeight / dz;

    	std::cout << "i_start:" << i_start << "\n";   
   	 	std::cout << "i_end:" << i_end << "\n";       
    	std::cout << "j_start:" << j_start << "\n";  
   	 	std::cout << "j_end:" << j_end << "\n";         
   	 	std::cout << "k_end:" << k_end << "\n";       

   	 	std::cout << "i_cut_start:" << i_cut_start << "\n";  
    	std::cout << "i_cut_end:" << i_cut_end << "\n";      
    	std::cout << "j_cut_start:" << j_cut_start << "\n"; 
    	std::cout << "j_cut_end:" << j_cut_end << "\n";        
    	std::cout << "k_cut_end:" << k_cut_end << "\n"; 
		
		/// defining cells cut by building
		for (int k = 1; k < k_cut_end; k++){
			for (int j = j_cut_start; j < j_cut_end; j++){
				for (int i = i_cut_start; i < i_cut_end; i++){
	                icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
					icellflag[icell_cent] = 7;                         /// Set cell index flag to cut-cell
				}
			}
    	}

		/// defining building solid cells
		for (int k = 0; k < k_end; k++){
			for (int j = j_start; j < j_end; j++){
				for (int i = i_start; i < i_end; i++){
					icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
					icellflag[icell_cent] = 0;                         /// Set cell index flag to building
				}
			}
		}

		/// defining ground solid cells
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int icell_cent = i + j*(nx-1);
				icellflag[icell_cent] = 0.0;
			}
		}
 
		float ***x_cut, ***y_cut, ***z_cut;
		x_cut = new float** [numcell_cent]();
		y_cut = new float** [numcell_cent]();
		z_cut = new float** [numcell_cent]();

		for (int i = 0; i<numcell_cent; i++){
			x_cut[i] = new float* [6]();
			y_cut[i] = new float* [6]();
			z_cut[i] = new float* [6]();
			for (int j = 0; j<6; j++){
				x_cut[i][j] = new float [6]();
				y_cut[i][j] = new float [6]();
				z_cut[i][j] = new float [6]();
			}
		}
	
		int **num_points;       /// number of intersection points in each face of the cell
		num_points = new int* [numcell_cent]();
		for (int i=0; i<numcell_cent; i++){
			num_points[i] = new int [6]();
		}
		int icell_cut;          /// cut-cell index
		
		/// intersection points hard-coded	
		for (int j=j_cut_start+1; j<j_cut_end; j++){
			for (int k=1; k<k_cut_end; k++){
				icell_cut = i_cut_start + j*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][2] = num_points[icell_cut][3] = num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][1] = 0;

					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;

					x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = x_start-i_cut_start*dx;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz;

					x_cut[icell_cut][4][0] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = x_start-i_cut_start*dx;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;
				
				}
				icell_cut = (i_cut_end-1) + j*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] = num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][0] = 0;

					y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
					y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
					z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
					z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;
	
					x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = dx;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][0] = x_cut[icell_cut][3][1] = x_cut[icell_cut][3][0] = x_start+L-i_end*dx;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][1] = z_cut[icell_cut][3][2] = z_cut[icell_cut][3][1] = 0.0;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz;
	
					x_cut[icell_cut][4][2] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = dx;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][0] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][0] = x_start+L-i_end*dx;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][2] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][0] = y_cut[icell_cut][5][3] = dy;
				}
			}
		}

		for (int i=i_cut_start+1; i<i_cut_end; i++){
			for (int k=1; k<k_cut_end; k++){
				icell_cut = i + j_cut_start*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][3] = 0;
	
					x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz;
	
					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = y_start-j_cut_start*dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;
	
					x_cut[icell_cut][4][0] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = y_start-j_cut_start*dy;
				}
				icell_cut = i + (j_cut_end-1)*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][3] = num_points[icell_cut][4] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][2] = 0;
	
					x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
					x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
					z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
					z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz;
	
					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = y_start+W-j_end*dy;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;
	
					x_cut[icell_cut][4][0] = x_cut[icell_cut][4][3] = x_cut[icell_cut][5][1] = x_cut[icell_cut][5][0] = 0.0;
					x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_cut[icell_cut][5][2] = x_cut[icell_cut][5][3] = dx;
					y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_cut[icell_cut][5][1] = y_cut[icell_cut][5][2] = y_start+W-j_end*dy;
					y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = y_cut[icell_cut][5][3] = y_cut[icell_cut][5][0] = dy;
				}
			}
		}
		
		for (int i=i_cut_start+1; i<i_cut_end; i++){
			for (int j=j_cut_start+1; j<j_cut_end; j++){
				icell_cut = i + j*(nx-1) + k_end*(nx-1)*(ny-1);
				if (icellflag[icell_cut]==7){
					num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] = num_points[icell_cut][5] = 4;
					num_points[icell_cut][4] = 0;

					x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
					x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
					y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
					y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;
	
					y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
					y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
					z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = H-(k_end-1)*dz;
					z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;
	
					x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
					x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
					z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = H-(k_end-1)*dz;
					z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz;
				}
			}
		}	
	
		for (int k=1; k<k_end; k++){		
			icell_cut = i_cut_start + j_cut_start*(nx-1) + k*(nx-1)*(ny-1);
			if (icellflag[icell_cut]==7){
				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] = 4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = y_start-j_cut_start*dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz;
			
				x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = x_start-i_cut_start*dx;
				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz;
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

				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] = 4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;
	
				y_cut[icell_cut][0][0] = y_cut[icell_cut][0][1] = 0.0;
				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = y_start-j_cut_start*dy;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;
				z_cut[icell_cut][0][1] = z_cut[icell_cut][0][2] = 0.0;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;
				
				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][1] = 0.0;
				x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = dx;
				z_cut[icell_cut][2][1] = z_cut[icell_cut][2][2] = 0.0;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = dz;

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][1] = x_start+L-i_end*dx;
				x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = dx;
				z_cut[icell_cut][3][1] = z_cut[icell_cut][3][2] = 0.0;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz;

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


				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] = 4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;

				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = y_start+W-j_end*dy;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;

				x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = x_start-i_cut_start*dx;
				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = dz;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][1] = 0.0;

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz;

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


				num_points[icell_cut][0] = num_points[icell_cut][1] = num_points[icell_cut][2] = num_points[icell_cut][3] = 4;
				num_points[icell_cut][5] = num_points[icell_cut][4] = 6;
	
				y_cut[icell_cut][0][0] = y_cut[icell_cut][0][1] = y_start+W-j_end*dy;
				y_cut[icell_cut][0][2] = y_cut[icell_cut][0][3] = dy;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;
				z_cut[icell_cut][0][1] = z_cut[icell_cut][0][2] = 0.0;

				y_cut[icell_cut][1][2] = y_cut[icell_cut][1][3] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;
				
				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][1] = x_start+L-i_end*dx;
				x_cut[icell_cut][2][2] = x_cut[icell_cut][2][3] = dx;
				z_cut[icell_cut][2][1] = z_cut[icell_cut][2][2] = 0.0;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][3] = dz;

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][1] = 0.0;
				x_cut[icell_cut][3][2] = x_cut[icell_cut][3][3] = dx;
				z_cut[icell_cut][3][1] = z_cut[icell_cut][3][2] = 0.0;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][3] = dz;

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
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz;
			
				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;
	
				y_cut[icell_cut][1][4] = y_cut[icell_cut][1][5] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				y_cut[icell_cut][1][3] = y_cut[icell_cut][1][2] = y_start-j_cut_start*dy;
				z_cut[icell_cut][1][4] = z_cut[icell_cut][1][3] = H-(k_end-1)*dz;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = 0.0;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][5] = dz;
		
				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][5] = 0.0;
				x_cut[icell_cut][3][3] = x_cut[icell_cut][3][4] = dx;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = x_start-i_cut_start*dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = H-(k_end-1)*dz;
				z_cut[icell_cut][3][4] = z_cut[icell_cut][3][5] = dz;

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
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;

				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][3] = 0.0;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = dz;
				
				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;
	
				y_cut[icell_cut][0][4] = y_cut[icell_cut][0][5] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				y_cut[icell_cut][0][3] = y_cut[icell_cut][0][2] = y_start-j_cut_start*dy;
				z_cut[icell_cut][0][4] = z_cut[icell_cut][0][3] = H-(k_end-1)*dz;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = 0.0;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][5] = dz;
	
				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][5] = 0.0;
				x_cut[icell_cut][3][3] = x_cut[icell_cut][3][4] = dx;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = x_start+L-i_end*dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = H-(k_end-1)*dz;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = 0.0;
				z_cut[icell_cut][3][4] = z_cut[icell_cut][3][5] = dz;

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
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][3] = dz;
	
				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz;
			
				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;
		
				y_cut[icell_cut][1][4] = y_cut[icell_cut][1][5] = dy;
				y_cut[icell_cut][1][1] = y_cut[icell_cut][1][0] = 0.0;
				y_cut[icell_cut][1][3] = y_cut[icell_cut][1][2] = y_start+W-j_end*dy;
				z_cut[icell_cut][1][4] = z_cut[icell_cut][1][3] = 0.0;
				z_cut[icell_cut][1][2] = z_cut[icell_cut][1][1] = H-(k_end-1)*dz;
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][5] = dz;
	
				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][5] = 0.0;
				x_cut[icell_cut][2][3] = x_cut[icell_cut][2][4] = dx;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_start-i_cut_start*dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = 0.0;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = H-(k_end-1)*dz;
				z_cut[icell_cut][2][4] = z_cut[icell_cut][2][5] = dz;

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
				z_cut[icell_cut][1][0] = z_cut[icell_cut][1][3] = dz;

				x_cut[icell_cut][3][0] = x_cut[icell_cut][3][3] = 0.0;
				x_cut[icell_cut][3][1] = x_cut[icell_cut][3][2] = dx;
				z_cut[icell_cut][3][0] = z_cut[icell_cut][3][1] = 0.0;
				z_cut[icell_cut][3][2] = z_cut[icell_cut][3][3] = dz;
			
				x_cut[icell_cut][5][0] = x_cut[icell_cut][5][3] = 0.0;
				x_cut[icell_cut][5][1] = x_cut[icell_cut][5][2] = dx;
				y_cut[icell_cut][5][0] = y_cut[icell_cut][5][1] = 0.0;
				y_cut[icell_cut][5][2] = y_cut[icell_cut][5][3] = dy;
		
				y_cut[icell_cut][0][4] = y_cut[icell_cut][0][5] = dy;
				y_cut[icell_cut][0][1] = y_cut[icell_cut][0][0] = 0.0;
				y_cut[icell_cut][0][3] = y_cut[icell_cut][0][2] = y_start+W-j_end*dy;
				z_cut[icell_cut][0][4] = z_cut[icell_cut][0][3] = 0.0;
				z_cut[icell_cut][0][2] = z_cut[icell_cut][0][1] = H-(k_end-1)*dz;
				z_cut[icell_cut][0][0] = z_cut[icell_cut][0][5] = dz;
	
				x_cut[icell_cut][2][0] = x_cut[icell_cut][2][5] = 0.0;
				x_cut[icell_cut][2][3] = x_cut[icell_cut][2][4] = dx;
				x_cut[icell_cut][2][1] = x_cut[icell_cut][2][2] = x_start+L-i_end*dx;
				z_cut[icell_cut][2][0] = z_cut[icell_cut][2][1] = H-(k_end-1)*dz;
				z_cut[icell_cut][2][2] = z_cut[icell_cut][2][3] = 0.0;
				z_cut[icell_cut][2][4] = z_cut[icell_cut][2][5] = dz;

				x_cut[icell_cut][4][0] = x_cut[icell_cut][4][5] = 0.0;
				x_cut[icell_cut][4][1] = x_cut[icell_cut][4][2] = x_start+L-i_end*dx;
				x_cut[icell_cut][4][3] = x_cut[icell_cut][4][4] = dx;
				y_cut[icell_cut][4][0] = y_cut[icell_cut][4][1] = y_start+W-j_end*dy;
				y_cut[icell_cut][4][2] = y_cut[icell_cut][4][3] = 0.0;
				y_cut[icell_cut][4][4] = y_cut[icell_cut][4][5] = dy;
			}


		float **coeff;				/// area fraction coeeficient
		coeff = new float* [numcell_cent]();
		for (int i=0; i<numcell_cent; i++){
			coeff[i] = new float [6]();
		}
	
    	for ( int k = 1; k < nz-2; k++){
    	    for (int j = 1; j < ny-2; j++){
    	        for (int i = 1; i < nx-2; i++){
					icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
	
					if (icellflag[icell_cent]==7){
						for (int ii=0; ii<6; ii++){
							coeff[icell_cent][ii] = 0;
							if (num_points[icell_cent][ii] !=0){
								/// calculate area fraction coeeficient for each face of the cut-cell
								for (int jj=0; jj<num_points[icell_cent][ii]-1; jj++){
									coeff[icell_cent][ii] += (0.5*(y_cut[icell_cent][ii][jj+1]+y_cut[icell_cent][ii][jj])*(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dy*dz) + (0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dx*dz) + (0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*(y_cut[icell_cent][ii][jj+1]-y_cut[icell_cent][ii][jj]))/(dx*dy);
								}

								coeff[icell_cent][ii] += (0.5*(y_cut[icell_cent][ii][0]+y_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dy*dz) + (0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dz) + (0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*(y_cut[icell_cent][ii][0]-y_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dy);

							}
						}

					/// Assign solver coefficients
					f[icell_cent] = coeff[icell_cent][0];
					e[icell_cent] = coeff[icell_cent][1];
					h[icell_cent] = coeff[icell_cent][2];
					g[icell_cent] = coeff[icell_cent][3];
					n[icell_cent] = coeff[icell_cent][4];
					m[icell_cent] = coeff[icell_cent][5];
					}	
			
					if (icellflag[icell_cent] !=0) {
					
						/// Wall bellow
						if (icellflag[icell_cent-(nx-1)*(ny-1)]==0) {
			    			n[icell_cent] = 0.0; 

						}
						/// Wall above
						if (icellflag[icell_cent+(nx-1)*(ny-1)]==0) {
			    			m[icell_cent] = 0.0;
						}
						/// Wall in back
						if (icellflag[icell_cent-1]==0){
							f[icell_cent] = 0.0; 
						}
						/// Wall in front
						if (icellflag[icell_cent+1]==0){
							e[icell_cent] = 0.0; 
						}
						/// Wall on right
						if (icellflag[icell_cent-(nx-1)]==0){
							h[icell_cent] = 0.0;
						}
						/// Wall on left
						if (icellflag[icell_cent+(nx-1)]==0){
							g[icell_cent] = 0.0; 
						}
					}				 
				}
			}    
		}	

	}
=======
		parsePrimitive<float>(true, height, "height");
		parsePrimitive<float>(true, baseHeight, "baseHeight");
		parsePrimitive<float>(true, centroidX, "centroidX");
		parsePrimitive<float>(true, centroidY, "centroidY");
		parsePrimitive<float>(true, xFo, "xFo");
		parsePrimitive<float>(true, yFo, "yFo");
		parsePrimitive<float>(true, length, "length");
		parsePrimitive<float>(true, width, "width");
		parsePrimitive<float>(true, rotation, "rotation");
		parsePrimitive<int>(false, buildingDamage, "buildingDamage");
		parsePrimitive<float>(false, atten, "atten");
		Wt = 0.5 * width;
		Lt = 0.5 * length;
	}

	void setBoundaries(float dx, float dy, float dz, int nz, float *zm)
	{
		iStart = xFo / dx + 1;  
		iEnd = (xFo + length) / dx;  
		jEnd = (yFo + width) / dy;  
		jStart = (yFo - width) / dy + 1;
/*		if (buildingDamage != 2)
		{
			for (int i = 1; i < nz - 1; i++)
			{
				kStart = i;
				if (baseHeightActual <= zm[i])
					break; 
			}
			for (int i = kStart; i < nz - 1; i++)
			{
				kEnd =i;
				if (height < zm[i + 1])
					break;
			}
		}*/
		kStart = baseHeight / dz;
		kEnd = kStart + (height / dz);
	}




>>>>>>> origin/doxygenAdd

/*
Note: The select case portion is almost identical across geometry types 1, 2, and 6.
Difference is that 6 has one more case, type 5. similar to default except sets to 1
instead of 0. Also same as 
*/

	void setCells(int nx, int ny, int nz, int *icellflag, int *ibldflag, int ibuild) 
	{
<<<<<<< HEAD
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

};

=======
		if(!rotation)
		{

		 for (int j = jStart; j <= jEnd; j++)
		 {
		    for (int i = iStart; i <= iEnd; i++)
		     {
		       switch (buildingType)
		       {
		          case 0:
		         	for ( int k = kStart; k <= kEnd; k++)
		         	{
		             icellflag[CELL(i,j,k,1)] = 1;
		             ibldflag[CELL(i,j,k,1)] = ibuild; //take in as parameter maybe?
		         	}
		         	break;
		          case 2:
		             for( int k = kStart; k <= kEnd; k++)
		             /*{
		                if( icellflag[i][j][k] != 0)
		                {
		                   if(lu_canopy_flag > 0)
		                   {
		                      if(canopy_top[i][j] == landuse_height[i][j])
		                      {
		                         if(k == 1) 
		                         	for (int c = 0; c < nz; c++)
		                         		canopy_atten[i][j][c] = 0.0f;
		                         if( height < 0.5f * dz_array[0])
		                            canopy_top[i][j] = 0.0f;
		                         else
		                         {
		                            canopy_top(i,j)=Ht(ibuild)
		                            canopy_atten(i,j,k)=atten(ibuild)
		                         }
		                      }
		                      else if( height > canopy_top[i][j])
		                         canopy_top[i][j] = height;
		                   }
		                   else
		                   {
		                      if(height > canopy_top[i][j])
		                         canopy_top[i][j] = height;
		                      canopy_atten[i][j][k] = atten;
		                   }
		                }
		             }*/
		             break;
		          case 3:
		             /* I don't really understand this part right now. ilevel and ceiling haven't been brought up before.
		             ilevel=0
		             do k=kstart(ibuild),kend(ibuild)
		                ilevel=ilevel+1
		                if(ilevel/2 .ne. ceiling(0.5*real(ilevel)))cycle
		                icellflag(i,j,k)=0
		                ibldflag(i,j,k)=ibuild
		             enddo*/
		             break;                             
		          default:
		         	for ( int k = kStart; k <= kEnd; k++)
		         	{
		             icellflag[CELL(i,j,k,1)] = 0;
		             ibldflag[CELL(i,j,k,1)] = ibuild; //take in as parameter maybe?
		         	}
		       }
		    }
		 }
		}
	}
};
/*                else
                {

//! calculate corner coordinates of the building
                     float x1 = xfo + width * sin(gamma);
                     float y1 = yfo(ibuild)-Wt(ibuild)*cos(gamma(ibuild))
                     float x2 = x1+Lti(ibuild)*cos(gamma(ibuild))
                     float y2 = y1+Lti(ibuild)*sin(gamma(ibuild))
                     float x4 = xfo(ibuild)-Wt(ibuild)*sin(gamma(ibuild))
                     float y4 = yfo(ibuild)+Wt(ibuild)*cos(gamma(ibuild))
                     float x3 = x4+Lti(ibuild)*cos(gamma(ibuild))
                     float y3 = y4+Lti(ibuild)*sin(gamma(ibuild))
 271                 format(8f8.3)
                     if(gamma(ibuild).gt.0)then
                        xmin=x4
                        xmax=x2
                        ymin=y1
                        ymax=y3
                     endif
                     if(gamma(ibuild).lt.0)then
                        xmin=x1
                        xmax=x3
                        ymin=y2
                        ymax=y4
                     endif
                     istart(ibuild)=nint(xmin/dx)
                     iend(ibuild)=nint(xmax/dx)
                     jstart(ibuild)=nint(ymin/dy)
                     jend(ibuild)=nint(ymax/dy)
!erp  do k=int(zfo(ibuild)),kend(ibuild)  
!erp        do j=int(ymin),int(ymax)
!erp     do i=int(xmin),int(xmax)
!erp     x_c=real(i) !x coordinate to be checked
!erp     y_c=real(j) !y coordinate to be checked
! changed int to nint in next three lines 8-14-06
                     do j=nint(ymin/dy)+1,nint(ymax/dy)+1   !convert back to real world unit, TZ 10/29/04
                        do i=nint(xmin/dx)+1,nint(xmax/dx)+1   !convert back to real world unit, TZ 10/29/04
                           x_c=(real(i)-0.5)*dx !x coordinate to be checked   !convert back to real world unit, TZ 10/29/04
                           y_c=(real(j)-0.5)*dy !y coordinate to be checked   !convert back to real world unit, TZ 10/29/04
!calculate the equations of the lines making up the 4 walls of the
!building
						   if( x4 .eq. x1)x4=x4+.0001
                           slope = (y4-y1)/(x4-x1) !slope of L1
                           xL1 = x4 + (y_c-y4)/slope
                           if( x3 .eq. x2)x3=x3+.0001
                           slope = (y3-y2)/(x3-x2) !slope of L2
                           xL2 = x3 + (y_c-y3)/slope
                           if( x2 .eq. x1)x2=x2+.0001
                           slope = (y2-y1)/(x2-x1) !slope of L3
                           yL3 = y1 + slope*(x_c-x1)
                           if( x3 .eq. x4)x3=x3+.0001
                           slope = (y3-y4)/(x3-x4) !slope of L4
                           yL4 = y4 + slope*(x_c-x4)
                           if(x_c.gt.xL1.and.x_c.lt.xL2.and.y_c.gt.yL3.and.y_c.lt.yL4)then
                              select case(bldtype(ibuild))
                                 case(0)
                                    icellflag(i,j,kstart(ibuild):kend(ibuild))=1
                                    ibldflag(i,j,kstart(ibuild):kend(ibuild))=ibuild
                                 case(2)
                                    do k=kstart(ibuild),kend(ibuild)
                                       if(icellflag(i,j,k) .ne. 0)then
                                          if(lu_canopy_flag .gt. 0)then
                                             if(canopy_top(i,j) .eq. landuse_height(i,j))then
                                                if(k .eq. 2)canopy_atten(i,j,:)=0.
                                                if(Ht(ibuild) .lt. 0.5*dz_array(1))then
                                                   canopy_top(i,j)=0.
                                                else
                                                   canopy_top(i,j)=Ht(ibuild)
                                                   canopy_atten(i,j,k)=atten(ibuild)
                                                endif
                                             elseif(Ht(ibuild) .gt. canopy_top(i,j))then
                                                canopy_top(i,j)=Ht(ibuild)
                                             endif
                                          else
                                             if(Ht(ibuild) .gt. canopy_top(i,j))then
                                                canopy_top(i,j)=Ht(ibuild)
                                             endif
                                             canopy_atten(i,j,k)=atten(ibuild)
                                          endif
                                       endif
                                    enddo
                                 case(3)
                                    ilevel=0
                                    do k=kstart(ibuild),kend(ibuild)
                                       ilevel=ilevel+1
                                       if(ilevel/2 .ne. ceiling(0.5*real(ilevel)))cycle
                                       icellflag(i,j,k)=0
                                       ibldflag(i,j,k)=ibuild
                                    enddo                                 
                                 case default
                                    icellflag(i,j,kstart(ibuild):kend(ibuild))=0
                                    ibldflag(i,j,kstart(ibuild):kend(ibuild))=ibuild
                              endselect
                           endif
                        enddo
                     enddo
                  endif
! generate cylindrical buildings
! need to specify a and b as the major and minor axis of
! the ellipse
! xco and yco are the coordinates of the center of the ellipse
               }*/
>>>>>>> origin/doxygenAdd
