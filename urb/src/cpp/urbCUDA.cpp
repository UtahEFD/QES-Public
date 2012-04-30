#include "urbCUDA.h"

namespace QUIC 
{

////////////
// Public //
////////////
	void urbCUDA::solveUsingGS_RB(QUIC::urbModule* um) 
	{
		float old_omega_relax = um->omegarelax;
		um->setOmegaRelax(1.0);
		
		QUIC::urbCUDA::solveUsingSOR_RB(um);
		
		um->setOmegaRelax(old_omega_relax);
	}

	// \\todo Possibly push it entirely to CUDA kernel.
	void urbCUDA::solveUsingSOR_RB(QUIC::urbModule* um) 
	{	
		if(!um->quiet) {std::cout << "Solving with SOR Red-Black..." << std::flush;}
	
			um->stpwtchs->comput->start();
		QUIC::urbCUDA::firstIteration(um);

		if(um->iter_step > 1) 
		{
			QUIC::urbCUDA::iterate(um, um->iter_step - 1);
			QUIC::urbCUDA::checkForConvergence(um);
		}

		while(!um->converged && um->iteration < um->max_iterations)
		{
			QUIC::urbCUDA::iterate(um, um->iter_step);
			QUIC::urbCUDA::checkForConvergence(um);
		}
			um->stpwtchs->comput->stop();

		if(!um->quiet) {std::cout << "done." << std::endl;}

		QUIC::urbCUDA::calcVelocities(um);
	}

	void urbCUDA::solveWithDiffusion(QUIC::urbModule* um, int const& steps) 
	{
		um->qwrite("Solving with diffusion...");
		for(int i = 0; i < steps; i++) 
		{
			QUIC::urbCUDA::solveUsingSOR_RB(um);

				um->stpwtchs->diffus->start();
			cudaDiffusion(um->d_vels, um->d_typs, um->d_visc, um->dx, um->dy, um->dz);
			cudaThreadSynchronize();
				um->stpwtchs->diffus->stop();
		}
		um->qwrite("done.\n");
	}
	

/////////////
// Private //
/////////////
	bool urbCUDA::firstIteration(QUIC::urbModule* um) 
	{
		um->converged = false;
		um->iteration = 0;
		um->eps  = 1.;
		um->abse = 0.;

	// Divergence
			um->stpwtchs->diverg->start();
		//Setup d_r
		cudaDivergence(um->d_r, um->d_vels,	um->alpha1, um->dx, um->dy, um->dz);
		cudaThreadSynchronize();
			um->stpwtchs->diverg->stop();


	// Pre-calc Denominators
			um->stpwtchs->denoms->start();
  	// Setup denominator in denoms for omega, o, p and q.
    cudaCompDenoms(um->d_bndrs, um->omegarelax, um->A, um->B);
		cudaThreadSynchronize();
			um->stpwtchs->denoms->stop(); 

	// Initial error.
		QUIC::urbCUDA::iterate(um, 1);
		QUIC::urbCUDA::checkForConvergence(um);
		
		um->eps       = um->abse*pow(10., -um->residual_reduction);
		um->converged = false; // checkForConvergence gives true for first iteration.
		
		return false;
	}

	void urbCUDA::iterate(QUIC::urbModule* um, int const& times = 1) 
	{
		//iterate times times.
		unsigned goto_iter = um->iteration + times;
		
		for(; um->iteration < goto_iter; um->iteration++) 
		{			      
			cudaIterCmprssdRBSOR
			(
			  um->d_p1, um->d_p2_err, um->d_bndrs, um->d_r, 
			  um->omegarelax, um->one_less_omegarelax, 
				um->dx, um->A, um->B
			);
		}
		
		//set floor boundry -- slice1 -> slice0
		///* When using viewer_urb for debugging, 'may want the following commented.
		cudaMemcpy
		(
			&um->d_p1[0], 
			&um->d_p1[um->nx*um->ny], 
			um->nx*um->ny*sizeof(float), 
			cudaMemcpyDeviceToDevice
		);
		//*/
	}

	void urbCUDA::checkForConvergence(QUIC::urbModule* um) 
	{
		// Calculate error.
		//int slc_size = um->nx*um->ny;
		cudaAbsDiff(um->d_p2_err, um->d_p1, um->d_p2_err, um->getDomainSize());
		
		// Sum errors.
		cudaSum(um->d_p2_err, um->domain_size, um->d_abse);
		cudaMemcpy(&um->abse, um->d_abse, sizeof(float), cudaMemcpyDeviceToHost);
		um->abse /= um->domain_size;
		
		// For testing of a different sum kernel.
		// um->abse  = cudaSum_v2(um->d_p2_err, um->domain_size);
		// um->abse /= um->domain_size;

		// p1 -> p2.
		cudaMemcpy
		(
			um->d_p2_err, 
			um->d_p1, 
			um->domain_size*sizeof(float), 
			cudaMemcpyDeviceToDevice
		);

		// Check for convergence.
		if(um->abse < um->getErrorTolerance()) {um->converged = true;}	
		
		//std::cout << "um->abse = " << um->abse << std::endl;
	}
	
	void urbCUDA::calcVelocities(QUIC::urbModule* um) 
	{	
		um->qwrite("Calculating velocities...");
		um->stpwtchs->euler->start();

		cudaEuler
		(
			um->d_vels, um->d_typs, 
			um->d_p1, um->alpha1, um->alpha2, 
			um->dx, um->dy, um->dz
		);
		cudaThreadSynchronize();

		um->stpwtchs->euler->stop();
		um->qwrite("done.\n");
	}
}

