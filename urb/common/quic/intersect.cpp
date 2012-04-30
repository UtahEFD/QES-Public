#include "intersect.h"

namespace QUIC
{
	void intersect::street(celltypes typs)
	{
		int nx = typs.dim.x;
		int ny = typs.dim.y;
		int nz = typs.dim.z;

		int cell_row = nx;
		int cell_slc = nx*ny;
		int cell_dmn = nx*ny*nz;

		int istart = 0;
		int jstart = 0;

		bool NS_flag    = false;
		bool changeflag = false;

		bool* intersect      = new bool[cell_dmn];
		bool* intersect_1    = new bool[cell_dmn];
		bool* intersect_2    = new bool[cell_dmn];
		bool* intersect_1opp = new bool[cell_dmn];
		bool* intersect_2opp = new bool[cell_dmn];
		
		bool* E_W_flag = new bool[cell_dmn];
		bool* W_E_flag = new bool[cell_dmn];
		bool* N_S_flag = new bool[cell_dmn];
		bool* S_N_flag = new bool[cell_dmn];
	
		for(int c = 0; c < cell_dmn; c++)
		{
			intersect[c]      = false;
			intersect_1[c]    = intersect_2[c]    = false;
			intersect_1opp[c] = intersect_2opp[c] = false;
			
			E_W_flag[c] = W_E_flag[c] = false;
			N_S_flag[c] = S_N_flag[c] = false;
		}

		// sweep through (x) to find intersections	
		for(int k = 0; k < nz; k++)
		{
			for(int j = 0; j < ny; j++)
			{
			// SUP sweep through +x
				for(int i = 1; i < nx; i++)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
				// determine where the street interesection begins
					if
					(
						typs.c[cI - 1] == CANYON && 
						typs.c[cI]     != CANYON && 
						typs.c[cI]     != SOLID
					)
					{
						changeflag = true;
						istart = i;
					}
				// determine where the street intersection ends
					// run into another street canyon, building or free atm.
					if(changeflag && isCanyonBuildingOrFluidQ(typs.c[cI]))
					{
						changeflag = false;
					}
					intersect_1[cI] = changeflag;
				}
				// if we get to the end of a row and changeflag = 1, then no SI exists reset those 
				if(changeflag)
				{
					for(int i = istart; i <= nx; i++)
					{
						int ndx = k*cell_slc + j*cell_row + i;
						
						intersect_1[ndx] = false;	
					}					
				}
				// reset flag
				changeflag = false;

				// SUP sweep through -x
				for(int i = nx - 2; i >= 1; i--)
				{
					int cI = k*cell_slc + j*cell_row + i;

					// determine where the street interesection begins
					if
					(
						typs.c[cI + 1] == CANYON && 
						typs.c[cI]     != CANYON && 
						typs.c[cI]     != SOLID
					)
					{
						changeflag = true;
						istart     = i;
					}
					// determine where the street bool*intersection ends
					// run into another street canyon, building or free atm.
					if(changeflag && isCanyonBuildingOrFluidQ(typs.c[cI]))
					{
						changeflag = false;
					}
					intersect_1opp[cI] = changeflag;
				}
				// if we get to the end of a row and changeflag = 1, then no SI exists reset those 
				if(changeflag)
				{
					// intersect_1opp(nx-1:istart_intflag:-1,j,k) = false;
					for(int i = nx - 1; i >= istart; i--)
					{
						int ndx = k*cell_slc + j*cell_row + i;
						
						intersect_1[ndx] = false;	
					}		
				}
				// reset flag
				changeflag = false;
			}
		}


//  now sweep in the j direction
		changeflag = false;
	
		for(int k = 0; k < nz; k++)
		{
			for(int i = 0; i < nx; i++)
			{
				for(int j = 1; j < ny; j++)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
					if
					(
						typs.c[cI - cell_row] == CANYON && 
						typs.c[cI]            != CANYON && 
						typs.c[cI]            != SOLID
					)
					{
						changeflag = true;
						jstart = j;
					}
					// determine where the street intersection ends
					// run into another street canyon, building or free atm.
					if(changeflag && isCanyonBuildingOrFluidQ(typs.c[cI]))
					{
						changeflag = false;
					}
					intersect_2[cI] = changeflag;
				}
				// if we get to the end of a row and changeflag = 1, then no SI exists reset those 
				if(changeflag)
				{
					// SUP changed intersect_1 to _2
					//intersect_2(i,jstart_intflag:ny-1,k) = false;
					for(int j = jstart; j < ny; j++)
					{
						int ndx = k*cell_slc + j*cell_row + i;
						
						intersect_2[ndx] = false;	
					}
				}
				changeflag = false;

// SUP sweep through -y
				for(int j = ny - 2; j >= 1; j--)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
					if
					(
						typs.c[cI + cell_row] == CANYON && 
						typs.c[cI]            != CANYON && 
						typs.c[cI]            != SOLID
					)
					{
						changeflag = true;
						jstart     = j;
					}
					// determine where the street intersection ends
					// run into another street canyon, building or free atm.
					if(changeflag && isCanyonBuildingOrFluidQ(typs.c[cI]))
					{
						changeflag = false;
					}
					intersect_2opp[cI] = changeflag;
				}
		// if we get to the end of a row and changeflag = 1, then no SI exists reset those 
				if(changeflag)
				{
					//intersect_2opp(i,ny-1:jstart_intflag:-1,k) = false;
					for(int j = ny - 1; j >= jstart; j--)
					{
						int ndx = k*cell_slc + j*cell_row + i;
						
						intersect_2[ndx] = false;	
					}
				}
				changeflag = false;
			}
		}

		for(int c = 0; c < cell_dmn; c++)
		{
			if
			(
				(intersect_1[c] || intersect_1opp[c]) && 
				(intersect_2[c] || intersect_2opp[c])
			)
			{
				intersect[c] = true;
			}
		}

// SUP looking to make sure that there are street canyons on 2 or more adjacent sides
		for(int k = 0; k < nz; k++)
		{
			for(int j = 1; j < ny; j++)
			{
				NS_flag = false;
				
				for(int i = 1; i < nx; i++)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
					if(intersect[cI] && typs.c[cI - 1] == CANYON) 
					{
						NS_flag = true;
					}
					if(!intersect[cI])
					{
						NS_flag = false;
					}
					if(NS_flag)
					{
						E_W_flag[cI] = true;
					}
				}
				
				NS_flag = false;
				for(int i = nx - 1; i >= 1; i--)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
					if(intersect[cI] && typs.c[cI + 1] == CANYON)
					{
						NS_flag = true;
					}
					if(!intersect[cI])
					{
						NS_flag = false;
					}
					if(NS_flag) 
					{
						W_E_flag[cI] = true;
					}
				}
			}
		}

		for(int k = 0; k < nz; k++)
		{
			for(int i = 1; i < nx; i++)
			{
				NS_flag = false;
				
				for(int j = 1; j < ny; j++)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
					if(intersect[cI] && typs.c[cI - cell_row] == CANYON)
					{
						NS_flag = true;
					}
					if(!intersect[cI])
					{
						NS_flag = false;
					}
					if(NS_flag)
					{
						S_N_flag[cI] = true;
					}
				}
				NS_flag = false;
				for(int j = ny - 1; j >= 1; j--)
				{
					int cI = k*cell_slc + j*cell_row + i;
				
					if(intersect[cI] && typs.c[cI + cell_row] == CANYON)
					{
						NS_flag = true;
					}
					if(!intersect[cI])
					{
						NS_flag = false;
					}
					if(NS_flag)
					{
						N_S_flag[cI] = true;
					}
				}
			}
		}
	// reset flag
		
		for(int c = 0; c < cell_dmn; c++)
		{
			if
			(
				(E_W_flag[c] || W_E_flag[c]) && 
				(S_N_flag[c] || N_S_flag[c])
			)
			{
				typs.c[c] = INTERSECTION;
			}
		}
		
		delete [] intersect;
		delete [] intersect_1;
		delete [] intersect_2;
		delete [] intersect_1opp;
		delete [] intersect_2opp;
		
		delete [] E_W_flag;
		delete [] W_E_flag;
		delete [] N_S_flag;
		delete [] S_N_flag;
	}
	
	void intersect::poisson
	(
		velocities ntls, 
		boundaryMatrices const& bndrs, 
		celltypes const& typs,
		float const& dx, float const& dy, float const& dz,
		unsigned int iterations // = 10
	)
	{
		iterations = (0 < iterations && iterations < 100) ? iterations : 10 ;
			
		int nx = bndrs.dim.x;
		int ny = bndrs.dim.y;
		int nz = bndrs.dim.z;
		
		int cell_row = nx;
		int cell_slc = nx*ny;
		int cell_dmn = nx*ny*nz;
		
		int grid_row = ntls.dim.x;
		int grid_slc = ntls.dim.x*ntls.dim.y;
		
		float* ep = new float[cell_dmn];
		float* fp = new float[cell_dmn];
		float* gp = new float[cell_dmn];
		float* hp = new float[cell_dmn];
		float* mp = new float[cell_dmn];
		float* np = new float[cell_dmn];
		
		float dx_dx = dx*dx;
		float dmmy  = 0.;
		
		for(int c = 0; c < cell_dmn; c++)
		{
			QUIC::decodeBoundary(bndrs.cmprssd[c], ep[c], fp[c], gp[c], hp[c], mp[c], np[c], dmmy, dmmy, dmmy);
		
			ep[c] /= dx_dx;
			fp[c] /= dx_dx;
			gp[c] /= dx_dx;
			hp[c] /= dx_dx;
			mp[c] /= dx_dx;
			np[c] /= dx_dx;
			
			float comp_sum = ep[c] + fp[c] + gp[c] + hp[c] + mp[c] + np[c];
			
			ep[c] /= comp_sum; // ep=ep/(ep+fp+gp+hp+mp+np)
			fp[c] /= comp_sum; // fp=fp/(ep+fp+gp+hp+mp+np)
			gp[c] /= comp_sum; // gp=gp/(ep+fp+gp+hp+mp+np)
			hp[c] /= comp_sum; // hp=hp/(ep+fp+gp+hp+mp+np)
			mp[c] /= comp_sum; // mp=mp/(ep+fp+gp+hp+mp+np)
			np[c] /= comp_sum; // np=np/(ep+fp+gp+hp+mp+np)
		}

		//bool nan_bndr_flag = false;
		//bool nan_ntls_flag = false;

		// \\todo Is there a problem is a canyon is next to a boundary cell?

		for(unsigned int iter = 1; iter < iterations; iter++)
		{
    	for(int k = 1; k < nz; k++)
     	for(int j = 1; j < ny; j++)
   		for(int i = 1; i < nx; i++)
   		{
  			int cI = k*cell_slc + j*cell_row + i;
  			int vI = k*grid_slc + j*grid_row + i;
  			
  			int vI_pi = vI + 1;
  			int vI_mi = vI - 1;
  			int vI_pj = vI + grid_row;
  			int vI_mj = vI - grid_row;
  			int vI_pk = vI + grid_slc;
  			int vI_mk = vI - grid_slc;      			
  		/*
  			if
  			(
  				ep[cI] != ep[cI] || fp[cI] != fp[cI] ||
  				gp[cI] != gp[cI] || hp[cI] != hp[cI] ||
  				mp[cI] != mp[cI] || np[cI] != np[cI]
  			)
  			{
  				nan_bndr_flag = true;
  			}
  		*/
				if(typs.c[cI] == INTERSECTION)
				{
					if(typs.c[cI - 1] == INTERSECTION)
					{
						ntls.u[vI] = 
						(
							(ep[cI]*ntls.u[vI_pi] + fp[cI]*ntls.u[vI_mi]) +
							(gp[cI]*ntls.u[vI_pj] + hp[cI]*ntls.u[vI_mj]) +
							(mp[cI]*ntls.u[vI_pk] + np[cI]*ntls.u[vI_mk])
						);
					}
					if(typs.c[cI - cell_row] == INTERSECTION)
					{
						ntls.v[vI] = 
						(
							(ep[cI]*ntls.v[vI_pi] + fp[cI]*ntls.v[vI_mi]) + 
							(gp[cI]*ntls.v[vI_pj] + hp[cI]*ntls.v[vI_mj]) +
							(mp[cI]*ntls.v[vI_pk] + np[cI]*ntls.v[vI_mk])
						);
					}
					if(typs.c[cI - cell_slc] == INTERSECTION)
					{
						ntls.w[vI] = 
						(
							(ep[cI]*ntls.w[vI_pi] + fp[cI]*ntls.w[vI_mi]) +
							(gp[cI]*ntls.w[vI_pj] + hp[cI]*ntls.w[vI_mj]) +
							(mp[cI]*ntls.w[vI_pk] + np[cI]*ntls.w[vI_mk])
						);
					}
        }
      /*  
        if
        (
        	ntls.u[vI] != ntls.u[vI] || 
        	ntls.v[vI] != ntls.v[vI] || 
        	ntls.w[vI] != ntls.w[vI]
        )
        {
        	nan_ntls_flag = true;
        }
      */
      }
   	}
   	
   //	if(nan_bndr_flag) {std::cout << "nan found in boundaries during poisson." << std::endl;}
   //	if(nan_ntls_flag) {std::cout << "nan found in velocities during poisson." << std::endl;}
   	
   	delete [] ep;
		delete [] fp;
		delete [] gp;
		delete [] hp;
		delete [] mp;
		delete [] np;
	}
	
	bool intersect::isCanyonBuildingOrFluidQ(CellType& type)
	{
		return (type == CANYON || type == SOLID || type == FLUID);
	}
}

