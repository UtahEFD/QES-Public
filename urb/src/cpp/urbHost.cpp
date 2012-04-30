#include "urbHost.h"

namespace QUIC 
{

// Public //

	void urbHost::solveUsingGS_RB(QUIC::urbModule* um) 
	{
		float old_omega_relax = um->omegarelax;
		um->setOmegaRelax(1.f);
		
		QUIC::urbHost::solveUsingSOR_RB(um);
		
		um->setOmegaRelax(old_omega_relax);
	}

	// \\todo Possibly push it entirely to CUDA kernel.
	void urbHost::solveUsingSOR_RB(QUIC::urbModule* um) 
	{	
		um->qwrite("Solving with SOR Red-Black (host)...");
		um->stpwtchs->comput->start();

		QUIC::urbHost::firstIteration(um);

		if(um->iter_step > 1) 
		{
			QUIC::urbHost::iterate(um, um->iter_step - 1);
			QUIC::urbHost::checkForConvergence(um);
		}

		while(!um->converged && um->iteration < um->max_iterations)
		{
			QUIC::urbHost::iterate(um, um->iter_step);
			QUIC::urbHost::checkForConvergence(um);
		}

		um->stpwtchs->comput->stop();
		um->qwrite("done.\n");

		QUIC::urbHost::calcVelocities(um);
	}

	void urbHost::solveWithDiffusion(QUIC::urbModule* um, int const& steps) 
	{
		um->qwrite("Solving with diffusion (host)...");
		for(int i = 0; i < steps; i++) 
		{
			QUIC::urbHost::solveUsingSOR_RB(um);
			um->stpwtchs->diffus->start();

			//cudaDiffusion(um->d_vels, um->d_typs, um->d_visc, um->dx, um->dy, um->dz);
			//cudaThreadSynchronize();
			
			um->stpwtchs->diffus->stop();
		}
		um->qwrite("done.\n");
	}
	

// Private //

	bool urbHost::firstIteration(QUIC::urbModule* um) 
	{
		um->converged = false;
		um->iteration = 0;
		um->eps  = 1.f;
		um->abse = 0.f;

		um->qwrite("Calculating divergence matrix...");
		um->stpwtchs->diverg->start();
			
	//Setup d_r
		//Do divergence on host.
		float dx_inv = 1 / um->dx;
		float dy_inv = 1 / um->dy;
		float dz_inv = 1 / um->dz;
				
		int grid_row = um->gx;
		int grid_slc = um->gx*um->gy;
				
		for(int cI = 0; cI < (int) um->domain_size; cI++)
		{
		  int gI = cI + int(cI / um->nx) + grid_row * int(cI / (um->nx*um->ny));
		  
		  um->h_r[cI] = (-2.f*um->alpha1*um->alpha1) 
					          * 
					          (
						          dx_inv*(um->h_vels.u[gI + 1]        - um->h_vels.u[gI]) 
					          + dy_inv*(um->h_vels.v[gI + grid_row] - um->h_vels.v[gI]) 
					          + dz_inv*(um->h_vels.w[gI + grid_slc] - um->h_vels.w[gI])
					          );
		}

		um->stpwtchs->diverg->stop();
		um->qwrite("finished.\n");

    // Pre-calc Denominators
  	um->qwrite("Calculating denominators...");
		um->stpwtchs->denoms->start();
    // Setup denominator in denoms for omega, o, p and q.
  	// Do the denomintors on host.
  	
    float o, p, q; float dmmy;
    
    for(int cI = 0; cI < (int) um->domain_size; cI++)
    {
      decodeBoundary
      (
      	um->h_bndrs.cmprssd[cI], 
      	dmmy, dmmy, dmmy, dmmy, dmmy, dmmy, 
      	o, p, q
      );

	    um->h_bndrs.denoms[cI] = um->omegarelax / (2.f*(o + um->A*p + um->B*q));
		}
		
		um->stpwtchs->denoms->stop();
		um->qwrite("finished.\n");

	  um->qwrite("Iterating once and checking for convergence...");
		um->stpwtchs->comput->start();
  
  	// Initial error.
		QUIC::urbHost::iterate(um, 1);
		QUIC::urbHost::checkForConvergence(um);
		
		um->eps       = um->abse*pow(10.f, -um->residual_reduction);
		um->converged = false; // checkForConvergence gives true for first iteration.
		
		um->stpwtchs->comput->stop();
		um->qwrite("finished.\n");
		
		return um->converged;
	}

	void urbHost::iterate(QUIC::urbModule* um, int const& times = 1) 
	{
		//iterate times times.
		unsigned goto_iter = um->iteration + times;
		
		int row_sz = um->nx;
		int slc_sz = um->nx*um->ny;
		
		float* cStick = &um->h_p1[0];
		float* fStick = &um->h_p1[ row_sz];
		float* bStick = &um->h_p1[-row_sz];
		float* uStick = &um->h_p1[ slc_sz];
		float* dStick = &um->h_p1[-slc_sz];

    //Host Iterate RBSOR
		for(; um->iteration < goto_iter; um->iteration++) 
		{	
		  for(int pass = 0; pass < 2; pass++)		  
		  for(int cI = slc_sz; cI < (int) um->domain_size - slc_sz; cI++)
		  {
		  	// Decompress the boundry conditions
        int cmprssd = um->h_bndrs.cmprssd[cI];
		    float e, f, g, h, m, n, dmmy;
			  decodeBoundary(cmprssd, e, f, g, h, m, n, dmmy, dmmy, dmmy);
		  
		    //std::cout << "Pass:" << pass << std::flush;
		    //std::cout << " :: Index:" << cI << std::flush;
		    //std::cout << " :: cmprssd:" << cmprssd << std::flush;
		    //std::cout << " :: dPM(cmprssd):" << decodePassMask(cmprssd) << std::flush;
		    //std::cout << " :: dDM(cmprssd):" << decodeDomainMask(cmprssd) << std::flush;
		    //std::cout << " :: all: " << (decodePassMask(cmprssd) ^ pass && !decodeDomainMask(cmprssd)) << std::endl;
		  
			  if(decodePassMask(cmprssd) ^ pass && !decodeDomainMask(cmprssd))
		    {
			    //Do the SOR calculation.
			    um->h_p1[cI] =  um->h_bndrs.denoms[cI]
							    		    *
							    		    (
							    					        (e*cStick[cI + 1] + f*cStick[cI - 1])
							    			    + um->A*(g*fStick[cI]     + h*bStick[cI])
							    			    + um->B*(m*uStick[cI]     + n*dStick[cI]) 
							    			    - um->dx*um->dx*um->h_r[cI] 
							    		    )
							    		    +
							    		    um->one_less_omegarelax * um->h_p1[cI];
		    }
		  }
		}
		
		//set floor boundry -- slice1 -> slice0
		memcpy(&um->h_p1[0], &um->h_p1[slc_sz], slc_sz*sizeof(float)); 
	}

	void urbHost::checkForConvergence(QUIC::urbModule* um) 
	{
	  //um->qwrite("Checking for convergence...");
	
		// Calculate error.
		//Host find matrix differences.
		um->abse = 0.f;
		for(int cI = 0; cI < (int) um->domain_size; cI++)
		{
		  um->abse += fabs(um->h_p1[cI] - um->h_p2_err[cI]);
		}
		
		um->abse /= um->domain_size;

    // Copy p1 -> p2 
		memcpy(um->h_p2_err, um->h_p1, um->domain_size*sizeof(float));
    
		// Check for convergence.
		if(um->abse < um->getErrorTolerance()) {um->converged = true;}	
		
    //um->qwrite("done.\n");
	}
	
	void urbHost::calcVelocities(QUIC::urbModule* um) 
	{	
		um->qwrite("Calculating velocities...");
		um->stpwtchs->euler->start();

    // Host could use Fortran here. (Out of the RB woods.)
    um->qwrite("No host velocities calculated. Needs implementation...");
				
		um->stpwtchs->euler->stop();
		um->qwrite("done.\n");
	}
}

