/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb.
* Source: Adapted from turbulence.f90 in QUICurbv5.? Fortran code.
*/

#ifndef TURBULENCE_H
#define TURBULENCE_H

#include "../cpp/celltypes.h"
#include "quicloader/velocities.h"

extern "C" void showError(char const* loc);

namespace QUIC
{
	// Turbulence kernel for calculating shear and populating visc matrix.
	__global__ void k_turbulence_shear
	(
		float* d_visc, 
		velocities d_vels,
		float dx, float dy, float dz 
	) 
	{
		int stick_length = blockDim.x;

		int nx = d_vels.dim.x - 1;
				
		int grid_slice = d_vels.dim.x*d_vels.dim.y;
		int grid_row   = d_vels.dim.x;

		int bidr = (int) blockIdx.x / (nx / stick_length);	
		int bidc =  blockIdx.x % (nx / stick_length);

		int gI = blockIdx.y * grid_slice + bidr * grid_row + bidc * stick_length + threadIdx.x;

		if(gI < grid_slice) // Zero the first slice.
		{
			d_visc[gI] = 0.f;
			return;
		}

		if(gI % grid_row == 0) //Don't do first col. Last column left out.
		{
			d_visc[gI] = 0.f;
			//d_visc[gI + grid_row] = 0.0;
			return;
		}

		if(bidr == 0) //Don't do first row. Last row left out.
		{
			d_visc[gI] = 0.f;
			//d_visc[gI + ny * grid_row] = 0.0;
			return;
		}

		float cs_les   = .2f;
		float mol_visc = .0000018f;
		float delta    = pow( dx * dy * dz, (1.f/3.f) );
		float dxi      = 1.f / dx;
		float dyi      = 1.f / dy;
		float dzi      = 1.f / dz;

		int gI_mi = gI - 1;
		int gI_pi = gI + 1;
		int gI_mj = gI - grid_row;
		int gI_pj = gI + grid_row;
		int gI_mk = gI - grid_slice;
		int gI_pk = gI + grid_slice;
		
		int gI_mi_pj = gI_mi + grid_row;
		int gI_mi_mj = gI_mi - grid_row;
		int gI_pi_mj = gI_pi - grid_row;
		int gI_pi_mk = gI_pi - grid_slice;
		
		int gI_mi_pk = gI_mi + grid_slice;
		int gI_mi_mk = gI_mi - grid_slice;

		int gI_mj_pk = gI_mj + grid_slice;
		int gI_mj_mk = gI_mj - grid_slice;
		int gI_pj_mk = gI_pj - grid_slice;

		float* d_u = d_vels.u;
		float* d_v = d_vels.v;
		float* d_w = d_vels.w;

		float shear = 2.f * 
					(
						  pow(dxi*(d_u[gI] - d_u[gI_mi]), 2.f)
					  + pow(dyi*(d_v[gI] - d_v[gI_mj]), 2.f)
					  + pow(dzi*(d_w[gI] - d_w[gI_mk]), 2.f)
					);
		shear += .25f * 
					(
						  pow(dyi*(d_u[gI_pj   ] - d_u[gI]      ) + dxi*(d_v[gI_pi   ] - d_v[gI      ]), 2.f)
					  + pow(dyi*(d_u[gI      ] - d_u[gI_mj]   ) + dxi*(d_v[gI_pi_mj] - d_v[gI_mj   ]), 2.f)
					  + pow(dyi*(d_u[gI_mi_pj] - d_u[gI_mi]   ) + dxi*(d_v[gI      ] - d_v[gI_mi   ]), 2.f)
					  + pow(dyi*(d_u[gI_mi   ] - d_u[gI_mi_mj]) + dxi*(d_v[gI_mj   ] - d_v[gI_mi_mj]), 2.f)
					);
		
		shear += .25f * 
					(
						  pow(dzi*(d_u[gI_pk   ] - d_u[gI      ]) + dxi*(d_w[gI_pi   ] - d_w[gI      ]), 2.f)
					  + pow(dzi*(d_u[gI      ] - d_u[gI_mk   ]) + dxi*(d_w[gI_pi_mk] - d_w[gI_mk   ]), 2.f)
					  + pow(dzi*(d_u[gI_mi_pk] - d_u[gI_mi   ]) + dxi*(d_w[gI      ] - d_w[gI_mi   ]), 2.f)
					  + pow(dzi*(d_u[gI_mi   ] - d_u[gI_mi_mk]) + dxi*(d_w[gI_mk   ] - d_w[gI_mi_mk]), 2.f)
					);
		shear += .25f * 
					(
						  pow(dzi*(d_v[gI_pk   ] - d_v[gI      ]) + dyi*(d_w[gI_pj   ] - d_w[gI      ]), 2.f)
					  + pow(dzi*(d_v[gI      ] - d_v[gI_mk   ]) + dyi*(d_w[gI_pj_mk] - d_w[gI_mk   ]), 2.f)
					  + pow(dzi*(d_v[gI_mj_pk] - d_v[gI_mj   ]) + dyi*(d_w[gI      ] - d_w[gI_mj   ]), 2.f)
					  + pow(dzi*(d_v[gI_mj   ] - d_v[gI_mj_mk]) + dyi*(d_w[gI_mk   ] - d_w[gI_mj_mk]), 2.f)
					);
		
		// Shared memory was overflowed before limiting threads to 192.
		d_visc[gI] = (cs_les * delta) * (cs_les * delta) * sqrt(fabs(shear)) + mol_visc;
	}

	// Kernel zeros visc matrix given corresponding surrounding building flags.
	__global__ void k_turbulence_building_flags(float* d_visc, celltypes d_typs) 
	{
		int nx = d_typs.dim.x;
		int ny = d_typs.dim.y;
	
		int stick_length = blockDim.x;
		int cell_slice =  nx * ny;
		int grid_row   =  nx + 1;
		int grid_slice = (nx + 1) * (ny + 1);

		int bidr = (int) blockIdx.x / (nx / stick_length);	
		int bidc =  blockIdx.x % (nx / stick_length);

		int gI = (blockIdx.y + 1) * grid_slice + bidr * grid_row + bidc * stick_length + threadIdx.x;
		int cI = (blockIdx.y + 1) * cell_slice + blockIdx.x * stick_length + threadIdx.x;

		if(cI % nx ==    0) return; //Don't modify first column.
		if(cI % nx == nx-1) return; //Don't modify last column.

		if(bidr ==    0) return; //Don't modify first row.
		if(bidr == ny-1) return; //Don't modify last row.

		if(d_typs.c[cI + 1]          == SOLID) d_visc[gI] = 0.f;
		if(d_typs.c[cI - 1]          == SOLID) d_visc[gI] = 0.f;
		if(d_typs.c[cI + nx]         == SOLID) d_visc[gI] = 0.f;
		if(d_typs.c[cI - nx]         == SOLID) d_visc[gI] = 0.f;
		if(d_typs.c[cI + cell_slice] == SOLID) d_visc[gI] = 0.f;
		if(d_typs.c[cI - cell_slice] == SOLID) d_visc[gI] = 0.f;
	}


	// Turbulence subroutine. Called from diffusion.
	extern "C"
	void cudaTurbulenceModel
	(
		float* d_visc, 
		celltypes d_typs, velocities d_vels,
		float dx, float dy, float dz 
	) 
	{
		int nx = d_typs.dim.x;
		int ny = d_typs.dim.y;
		int nz = d_typs.dim.z;

		// Using 192 here because even though we don't assign any shared memory,
		// 256 threads will use more memory than a given block has available.
		// Needs to change to improve performace. Get largest multiple of 64 that
		// divides threads. 
		// Want this to be a multiple of 64.
		int threads = (nx > 192) ? 64 : nx ;  

		int blocks_per_slice = (nx * ny) / (threads);
		int slices = nz;

		dim3 dimGrid(blocks_per_slice, slices);
		dim3 dimBlock(threads);	

		k_turbulence_shear<<< dimGrid, dimBlock >>>(d_visc, d_vels, dx, dy, dz); 
		showError("turbulence_shear");

		/* 
			!!! This cudaMemcpy's are to get the job done, not to be clean,
			!!! fast, proper or any other desirable thing. There has to be
			!!! a better way. Should probably write a kernel for all this.
			!!! // \\todo ...Josh?...
		*/

		int gx = d_vels.dim.x;
		int gy = d_vels.dim.y;
		int gz = d_vels.dim.z;

		int grid_slc = gx*gy;
		int grid_row   = gx;

		//visc(1,:,:)=visc(nx-1,:,:)
		for(int k = 0; k < gz; k++) 
		{
			for(int j = 0; j < gy; j++) 
			{
				cudaMemcpy
				(
					&d_visc[k*grid_slc + j*grid_row +      0],
					&d_visc[k*grid_slc + j*grid_row + nx - 1],
					sizeof(float),
					cudaMemcpyDeviceToDevice
				);
			}
		}
		//visc(nx,:,:)=visc(2,:,:)
		for(int k = 0; k < gz; k++) 
		{
			for(int j = 0; j < gy; j++) 
			{
				cudaMemcpy
				(
					&d_visc[k*grid_slc + j*grid_row + nx],
					&d_visc[k*grid_slc + j*grid_row +  1],
					sizeof(float),
					cudaMemcpyDeviceToDevice
				);
			}
		}
		
		//visc(:,:,1)=0.0 // Done. //

		//visc(:,:,nz)=visc(:,:,2)
		cudaMemcpy
		(
			&d_visc[nz*grid_slc], 
			&d_visc[1 *grid_slc], 
			grid_slc*sizeof(float), 
			cudaMemcpyDeviceToDevice
		);

		//visc(:,1,:)=visc(:,ny-1,:)
		for(int k = 0; k < gz; k++) 
		{
			cudaMemcpy
			(
				&d_visc[k*grid_slc +  	   0 *grid_row], 
				&d_visc[k*grid_slc + (ny - 1)*grid_row], 
				gx*sizeof(float), 
				cudaMemcpyDeviceToDevice
			);
		}
		//visc(:,ny,:)=visc(:,2,:)
		for(int k = 0; k < gz; k++) 
		{
			cudaMemcpy
			(
				&d_visc[k*grid_slc + ny*grid_row], 
				&d_visc[k*grid_slc +  1*grid_row], 
				gx*sizeof(float), 
				cudaMemcpyDeviceToDevice
			);
		}

		dimGrid = dim3(blocks_per_slice, slices - 2);
		k_turbulence_building_flags<<< dimGrid, dimBlock >>>(d_visc, d_typs); 
		showError("turbulence_building_flags");
		}
	}

#endif
