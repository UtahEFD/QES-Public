/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb
*/

#ifndef URBCUDA_H
#define URBCUDA_H

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include "urbModule.h"

extern "C" 
{
	void cudaAbsDiff(float* d_err, float* d_p1, float* d_p2, unsigned int size);

  void cudaCompDenoms
  (
  	QUIC::boundaryMatrices d_bndrs, 
  	float omegarelax, float A, float B
  );

	void cudaDivergence
	(
		float* d_r, QUIC::velocities d_vels,
		float alpha1, 
		float dx, float dy, float dz
	);

	void cudaDiffusion
	(
		QUIC::velocities d_vels, QUIC::celltypes d_typs, 
		float* d_visc, float dx, float dy, float dz
	);

	void cudaEuler
	(
		QUIC::velocities d_vels, QUIC::celltypes d_typs,  
		float* d_p1, float alpha1, float alpha2, 
		float dx, float dy, float dz
	);

	void cudaIterCmprssdRBSOR
	(
    float* d_p1, float* d_p2, 
    QUIC::boundaryMatrices d_bndrs, float* d_r, 
    float omegarelax, float one_less_omegarelax, 
    float A, float B, float dx
	);

	void cudaSum(float* d_err, int size, float* d_sum);
	//float cudaSum_v2(float* d_err, unsigned int size);
}

namespace QUIC 
{

	/**
	* Part of the QUIC namespace, urbCUDA is a stateless class that runs the 
	* necessary CUDA kernels on an urbModule. Uses CUDA to solve the SOR 
	* iteration technique found in sor3d.f90.
	* 
	* NOTES:
	* Currently, the divergence, sor3d, euler and diffusion calls are handled 
	* by this class. The other fortran functions needed for setup can be found in
	* urbSetup. 
	*/
	class urbCUDA 
	{
		
		friend class urbViewer;

		protected:

			static const int dflt_blck_sz = 64; // Link to kernel usage.

			/** Executes the first iteration. **/
			static bool firstIteration(QUIC::urbModule*);
			
			/**
			* Executes a specified number of iterations without checking for convergence.
			*
			* @param times is the number of iterations to execute. Default: 1.
			*/
			static void iterate(QUIC::urbModule*, int const& times);
			
			/**
			* Compares the current iterate with what's stored as the old iterate. 
			* The distance between the old iterate and the new iterate is not always one.
			* #NOTE: See iterate.
			*/
			static void checkForConvergence(QUIC::urbModule*);
			
			/**
			* Calls euler kernel, using the current lagragian field to calculate u, v and w.
			*/
			static void calcVelocities(QUIC::urbModule*);


		public:
				
			/**
			* Solves the problem using solveUsingSOR_RB with omegarelax set to 1.0.
			* Again the red black approach is required for implementation with parallel processing.
			*/
			static void solveUsingGS_RB(QUIC::urbModule*);
			
			/**
			* Solves the problem using a successive overrelaxation red black technique.
			* The red black portion of the technique is required for implementation using CUDA.
			*
			* 
			* @param omegarelax an important parameter for the SOR method. Default: 1.78.
			*/
			static void solveUsingSOR_RB(QUIC::urbModule*);
			
			/**
			* Solves the problem using diffusion after each solution is found using Red-Black SOR
			* technique and euler is applied. Then sor3d and euler are run again and diffusion
			* is applied. This is done steps times. Diffusion modifies initial
			* velocity matrices uo, vo and wo.
			*
			* @param step number of times to call diffusion and refind solution.
			*/
			static void solveWithDiffusion(QUIC::urbModule*, int const& steps);
	};
}

#endif
