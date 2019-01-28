
#include "Cut_cell.h"




void Cut_cell::calculateCoefficient(Cell* cells, const DTEHeightField* DTEHF, int nx, int ny, int nz, float dx, float dy,float dz,
								std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, 
								std::vector<float> &h, std::vector<float> &g, float pi, std::vector<int> &icellflag)
{


	int count = 0;
	std::vector<int> cutcell_index;
	std::vector< Vector3<float>> cut_points;
	Vector3 <float> location;


	cells = new Cell[(nx-1)*(ny-1)*(nz-1)];
	cutcell_index = DTEHF->setCells(cells, nx, ny, nz, dx, dy, dz);

	std::cout<<"number of cut cells:" << cutcell_index.size() << "\n";

	for (int i=0; i<nx-1; i++)
	{
		for (int j=0; j<ny-1; j++)
		{
			for (int k=0; k<nz-1; k++)
			{
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
				if (cells[icell_cent].getIsTerrain())
				{
					icellflag[icell_cent] = 0;
					//std::cout<<"cell Id:" << icell_cent << "\n";
				}
			}
		}
	} 

	//for all cut cells
	for (int j=0; j<cutcell_index.size(); j++)
	{
		//for every face
		for (int i=0; i<6; i++)
		{
			cut_points.clear();
			cut_points = cells[cutcell_index[j]].getFaceFluidPoints(i);
			location = cells[cutcell_index[j]].getLocationPoints();
			if (cut_points.size()<3 && cut_points.size()>0)
			{
				count += 1;
			}

			//if valid
			if (cut_points.size()>2)
			{
				//place points in local cell space
				for (int jj =0; jj<cut_points.size(); jj++)
				{
					for (int l=0; l<3; l++)
					{
						cut_points[jj][l] = cut_points[jj][l] - location[l];
					}

				}	
				reorderPoints(cut_points, i, pi);

				calculateArea(cut_points, cutcell_index[j], dx, dy, dz, n, m, f, e, h, g, i);	
			}
		}
	}

	std::cout<<"counter:" << count << "\n";

}



void Cut_cell::reorderPoints(std::vector< Vector3<float>> &cut_points, int index, float pi)
{

	Vector3<float> centroid;
	std::vector<float> angle (cut_points.size(), 0.0);
	Vector3<float> sum;

	sum[0] = 0;
	sum[1] = 0;
	sum[2] = 0;

	for (int i=0; i<cut_points.size(); i++)
	{
		sum[0] += cut_points[i][0];
		sum[1] += cut_points[i][1];
		sum[2] += cut_points[i][2];
	}

	centroid[0] = sum[0]/cut_points.size();
	centroid[1] = sum[1]/cut_points.size();
	centroid[2] = sum[2]/cut_points.size();
			
	for (int i=0; i<cut_points.size(); i++)
	{
		if (index==0 || index==1)
		{
			angle[i] = (180/pi)*atan2((cut_points[i][2]-centroid[2]),(cut_points[i][1]-centroid[1]));

		}
		if (index==2 || index==3)
		{
			angle[i] = (180/pi)*atan2((cut_points[i][2]-centroid[2]),(cut_points[i][0]-centroid[0]));
		}
		if (index==4 || index==5)
		{
			angle[i] = (180/pi)*atan2((cut_points[i][1]-centroid[1]),(cut_points[i][0]-centroid[0]));

		}
	}

	sort(angle, cut_points, pi);

}



void Cut_cell::sort(std::vector<float> &angle, std::vector< Vector3<float>> &cut_points, float pi)
{

	std::vector<float> angle_temp (cut_points.size(), 0.0);
	std::vector<float> angle_max (cut_points.size(), 0.0);
	std::vector<Vector3<float>> cut_points_temp;
	std::vector<int> imax (cut_points.size(),0);

	cut_points_temp = cut_points;

	for (int i=0; i<cut_points.size(); i++)
	{
		imax[i] = i;
		angle_max[i] = -180;
		angle_temp[i] = angle[i];
	}

	for (int i=0; i<cut_points.size(); i++)
	{
		for (int j=0; j<cut_points.size(); j++)
		{
			if (angle[j] > angle_max[i]){
				angle_max[i] = angle[j];
				imax[i] = j;
			}
		}
		angle[imax[i]] = -999;
	}

	for (int i=0; i<cut_points.size(); i++)
	{
		cut_points[i][0] = cut_points_temp[imax[cut_points.size()-1-i]][0];
		cut_points[i][1] = cut_points_temp[imax[cut_points.size()-1-i]][1];
		cut_points[i][2] = cut_points_temp[imax[cut_points.size()-1-i]][2];
		angle[i] = angle_temp[imax[cut_points.size()-1-i]];
	}

}
	

void Cut_cell::calculateArea(std::vector< Vector3<float>> &cut_points, int cutcell_index, float dx, float dy, float dz, 
							std::vector<float> &n, std::vector<float> &m, std::vector<float> &f, std::vector<float> &e, 
							std::vector<float> &h, std::vector<float> &g, int index)
{

	
	float coeff = 0;
	if (cut_points.size() !=0)
	{
	/// calculate area fraction coeeficient for each face of the cut-cell
		for (int i=0; i<cut_points.size()-1; i++)
		{
			coeff += (0.5*(cut_points[i+1][1]+cut_points[i][1])*(cut_points[i+1][2]-cut_points[i][2]))/(dy*dz) +
					 (0.5*(cut_points[i+1][0]+cut_points[i][0])*(cut_points[i+1][2]-cut_points[i][2]))/(dx*dz) + 
					 (0.5*(cut_points[i+1][0]+cut_points[i][0])*(cut_points[i+1][1]-cut_points[i][1]))/(dx*dy);
		}

		coeff += (0.5*(cut_points[0][1]+cut_points[cut_points.size()-1][1])*(cut_points[0][2]-
				 cut_points[cut_points.size()-1][2]))/(dy*dz) + (0.5*(cut_points[0][0]+cut_points[cut_points.size()-1][0])*
				 (cut_points[0][2]-cut_points[cut_points.size()-1][2]))/(dx*dz) + (0.5*(cut_points[0][0]+
				 cut_points[cut_points.size()-1][0])*(cut_points[0][1]-cut_points[cut_points.size()-1][1]))/(dx*dy);

	}
	if (coeff>=0){	
		if (index==0)
		{
			f[cutcell_index] = coeff;
		}
		if (index==1)
		{
			e[cutcell_index] = coeff;
		}
		if (index==2)
		{
			h[cutcell_index] = coeff;
		}
		if (index==3)
		{
			g[cutcell_index] = coeff;
		}
		if (index==4)
		{
			n[cutcell_index] = coeff;
		}
		if (index==5)
		{
			m[cutcell_index] = coeff;
		}
	}
}
