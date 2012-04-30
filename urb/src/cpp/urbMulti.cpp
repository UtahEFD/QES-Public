#include "urbMulti.h"

namespace QUIC
{

	void urbMulti::solveUsingMultiGPU(QUIC::urbModule& um, float omegarelax = 1.78) {QUIC::urbMulti::solveUsingMultiGPU(&um, omegarelax);}
	void urbMulti::solveUsingMultiGPU(QUIC::urbModule* um, float omegarelax = 1.78) 
	{
    Timer_t comput_start = um->timer.tic();

		int threads = (um->nx < 512) ? um->nx : threads = dflt_blck_sz ;

		um->setOmegaRelax(omegarelax); // Sets both omega and 1 - omega.
		/*
		QUIC::urbCUDA::firstIteration(um);

		if(um->iter_step > 1) 
		{
			QUIC::urbCUDA::iterate(um, um->iter_step - 1);
			QUIC::urbCUDA::checkForConvergence(um);
		}
		*/
		CUTThread GPUThreads[um->dvc_cnt];
		TGPUurbPlan GPUurbPlans[um->dvc_cnt];
		bool GPUconvergence[um->dvc_cnt];
		float* GPU_swp_sndrs[(um->dvc_cnt - 1) * 2];
		float* GPU_swp_rcvrs[(um->dvc_cnt - 1) * 2];
		int	GPU_swp_szs[(um->dvc_cnt - 1) * 2];

		int ttl_grp_cnt = um->domain_size / threads;

		for(int d = 0; d < um->dvc_cnt; d++)
		{
			GPUurbPlans[d].d_ID = d;
			GPUurbPlans[d].d_cnt = um->dvc_cnt;
			GPUurbPlans[d].threads = threads;
			GPUurbPlans[d].convergence = GPUconvergence;
			GPUurbPlans[d].d_swp_sndrs = GPU_swp_sndrs;
			GPUurbPlans[d].d_swp_rcvrs = GPU_swp_rcvrs;
			GPUurbPlans[d].swp_szs = GPU_swp_szs;
			GPUurbPlans[d].grp_ffst = BLOCK_LOW(d, um->dvc_cnt, ttl_grp_cnt);
			GPUurbPlans[d].grp_cnt = BLOCK_SIZE(d, um->dvc_cnt, ttl_grp_cnt);
			GPUurbPlans[d].um = um;

			//std::cout << "Device " << d << ":" << std::endl;
			//std::cout << "grp_ffst = " << GPUurbPlans[d].grp_ffst << std::endl;
			//std::cout << "grp_cnt = " << GPUurbPlans[d].grp_cnt << std::endl;
			//std::cout << "Check access to urbModule: " << std::endl;
			//std::cout << "GPUurbPlans[d].um->domain_size = " << GPUurbPlans[d].um->domain_size << std::endl;
		}

		for(int d = 0; d < um->dvc_cnt; d++)
		{
			GPUThreads[d] = cutStartThread((CUT_THREADROUTINE) threadedSORSolver, (void*) (GPUurbPlans + d));
		}

		cutWaitForThreads(GPUThreads, um->dvc_cnt);

		Timer_t comput_end = um->timer.tic();
		um->device_comput_time = um->timer.deltas(comput_start, comput_end);

		QUIC::urbCUDA::calcVelocities(um);
	}

	CUT_THREADPROC urbMulti::threadedSORSolver(TGPUurbPlan* p)
	{
		cudaSetDevice(p->d_ID); // Device is set. (Don't do it again?)
		showError("urbMulti::threadedSORSolver -- cudaSetDevice");

		pallocMulti(p);
		splitDomainMulti(p);
		setSwapPointersAndSizes(p);

		while(!p->um->converged && p->um->iteration < p->um->max_iterations)
		{
			iterateMulti(p, p->um->iter_step);
			checkForConvergenceMulti(p);
			// sync
			//swapDomainMulti(p);
		}

		//spliceDomainMulti(p);
	}

	void urbMulti::pallocMulti(TGPUurbPlan* p)
	{
		// Allocates memory on the currently selected GPU for the SOR needed data.
		// Allocates much more than needed (technically), but done this way as a
		// first pass to simplify the splitting, swapping and splicing.

		cudaMalloc((void**) &p->d_e, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_e");
		cudaMalloc((void**) &p->d_f, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_f");
		cudaMalloc((void**) &p->d_g, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_g");
		cudaMalloc((void**) &p->d_h, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_h");
		cudaMalloc((void**) &p->d_m, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_m");
		cudaMalloc((void**) &p->d_n, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_n");

		cudaMalloc((void**) &p->d_r, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_r");

		cudaMalloc((void**) &p->d_p1, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_p1");
		cudaMalloc((void**) &p->d_p2, p->um->domain_size * sizeof(float)); 		showError("urbMulti::pallocMulti -- allocating p->d_p2");

		cudaMalloc((void**) &p->d_err, p->um->domain_size * sizeof(float)); 	showError("urbMulti::pallocMulti -- allocating p->d_err");
		cudaMalloc((void**) &p->d_abse, sizeof(float));							showError("urbMulti::pallocMulti -- allocating p->d_abse");

		// May need to be changed in the future.
		cudaZero(p->d_p1, p->um->domain_size); 									showError("urbMulti::pallocMulti -- zeroing p->d_p1");
	}

	void urbMulti::splitDomainMulti(TGPUurbPlan* p)
	{
		// Do it stupid for everything but p1 and p2.
		cudaMemcpy(p->d_e, p->um->d_e, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_e");
		cudaMemcpy(p->d_f, p->um->d_f, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_f");
		cudaMemcpy(p->d_g, p->um->d_g, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_g");
		cudaMemcpy(p->d_h, p->um->d_h, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_h");
		cudaMemcpy(p->d_m, p->um->d_m, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_m");
		cudaMemcpy(p->d_n, p->um->d_n, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_n");

		cudaMemcpy(p->d_r, p->um->d_r, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);	showError("urbMulti::splitDomainMulti -- memcpy p->d_r");

		// Transfer info from device 0 to other devices along block allocation splitting scheme.
		//int lmnt_ffst = p->grp_ffst * p->threads;
		//cudaMemcpy(&p->d_p1[lmnt_ffst], &p->um->d_p1[lmnt_ffst], p->grp_cnt * p->threads * sizeof(float), cudaMemcpyDeviceToDevice);

		// Zeroed to start. Don't care to start with. Difficulty in splice...
		cudaMemcpy(p->d_p1, p->um->d_p1, p->um->domain_size * sizeof(float), cudaMemcpyDeviceToDevice);
		showError("urbMulti::splitDomainMulti -- memcpy p->d_p1");
	}

	void urbMulti::setSwapPointersAndSizes(TGPUurbPlan* p)
	{
		int grps_pr_slc = p->um->nx * p->um->ny / p->threads;

		int lwr_grp_slc_ndx = p->grp_ffst / grps_pr_slc; // The slice the current group is in.
		int lwr_bm_slc = lwr_grp_slc_ndx - 1;
		int lwr_tp_slc = (p->grp_ffst % grps_pr_slc == 0) ? lwr_grp_slc_ndx : lwr_grp_slc_ndx + 1 ;

		int lwr_bm_lmnt_ndx = lwr_bm_slc * grps_pr_slc * p->threads;
		int lwr_tp_lmnt_ndx = p->grp_ffst * grps_pr_slc * p->threads;

		int ppr_grp_slc_ndx = (p->grp_ffst + p->grp_cnt) / grps_pr_slc;
		int ppr_bm_slc = ppr_grp_slc_ndx - 1;
		int ppr_tp_slc = ((p->grp_ffst + p->grp_cnt) % grps_pr_slc == 0) ? ppr_grp_slc_ndx : ppr_grp_slc_ndx + 1 ;

		int ppr_bm_lmnt_ndx = ppr_bm_slc * grps_pr_slc * p->threads;
		int ppr_tp_lmnt_ndx = (p->grp_ffst + p->grp_cnt) * grps_pr_slc * p->threads;

		if(p->d_ID == 0)
		{
			// Upper Bottom Sender
			p->d_swp_sndrs[p->d_ID] = &p->d_p1[ppr_bm_lmnt_ndx];
			p->swp_szs    [p->d_ID]	= ppr_tp_lmnt_ndx - ppr_bm_lmnt_ndx;

			// Upper Top Receiver
			p->d_swp_rcvrs[p->d_ID + 1] = &p->d_p1[ppr_tp_lmnt_ndx];
		}
		else if(p->d_ID != p->d_cnt - 1)
		{
			// Lower Bottom Receiver
			p->d_swp_rcvrs[(p->d_ID - 1) * 2] =	&p->d_p1[lwr_bm_lmnt_ndx];

			// Lower Top Sender
			p->d_swp_sndrs[p->d_ID * 2 - 1] = &p->d_p1[lwr_tp_lmnt_ndx];
			p->swp_szs    [p->d_ID * 2 - 1] = ((lwr_tp_slc + 1) * grps_pr_slc - p->grp_ffst) * p->threads;

			// Upper Bottom Sender
			p->d_swp_sndrs[p->d_ID * 2]	= &p->d_p1[ppr_bm_lmnt_ndx];
			p->swp_szs    [p->d_ID * 2] = ppr_tp_lmnt_ndx - ppr_bm_lmnt_ndx;

			// Upper Top Receiver
			p->d_swp_rcvrs[p->d_ID] = &p->d_p1[ppr_tp_lmnt_ndx];
		}
		else // Last device
		{
			// Lower Bottom Receiver
			p->d_swp_rcvrs[(p->d_ID - 1) * 2] =	&p->d_p1[lwr_bm_lmnt_ndx];

			// Lower Top Sender
			p->d_swp_sndrs[p->d_ID * 2 - 1] = &p->d_p1[lwr_tp_lmnt_ndx];
			p->swp_szs    [p->d_ID * 2 - 1] = ((lwr_tp_slc + 1) * grps_pr_slc - p->grp_ffst) * p->threads;
		}		
		// Hopefully pointers and sizes are sorted as going through the domains, starting at 0,0,0.
	}

	void urbMulti::iterateMulti(TGPUurbPlan* p, int times) 
	{
		int lmnt_ffst = p->grp_ffst * p->threads;

		// Iterate time times.
		for(int i = 0; i < times; i++) 
		{
			//Remember: denominator is in e, f, g, ...
			cudaIterRBMulti
			(
				&p->d_e[lmnt_ffst], &p->d_f[lmnt_ffst], &p->d_g[lmnt_ffst], 
				&p->d_h[lmnt_ffst], &p->d_m[lmnt_ffst], &p->d_n[lmnt_ffst], 
				&p->d_r[lmnt_ffst], 
				p->um->one_less_omegarelax, 
				&p->d_p1[lmnt_ffst], &p->d_p2[lmnt_ffst], 
				p->um->nx, p->um->ny, p->grp_cnt, p->threads
			);

			//set floor boundry -- slice1 -> slice0
			if(p->d_ID == 0)
			{	
				int slc_sz = p->um->nx * p->um->ny;
				cudaMemcpy(&p->d_p1[0], &p->d_p1[slc_sz], slc_sz * sizeof(float), cudaMemcpyDeviceToDevice);
				showError("urbMulti::iterateMulti -- memcpy slice 1 -> slice 0");
				p->um->iteration++;
			}
		}
	}

	void urbMulti::checkForConvergenceMulti(TGPUurbPlan* p) 
	{
		int lmnt_ffst = p->grp_ffst * p->threads;
		int grp_lmnts = p->grp_cnt * p->threads;

		// Calculate error.
		cudaAbsDiff(&p->d_err[lmnt_ffst], &p->d_p1[lmnt_ffst], &p->d_p2[lmnt_ffst], grp_lmnts);
		
		// p1 -> p2.
		cudaMemcpy(&p->d_p2[lmnt_ffst], &p->d_p1[lmnt_ffst], grp_lmnts * sizeof(float), cudaMemcpyDeviceToDevice);
		showError("urbMulti::checkForConvergence -- memcpy p1 --> p2");

		// Sum errors.
		cudaSum(&p->d_err[lmnt_ffst], grp_lmnts, p->d_abse);

		float abse = 0.; cudaMemcpy(&abse, p->d_abse, sizeof(float), cudaMemcpyDeviceToHost);
		showError("urbMulti::checkForConvergence -- memcpy p->d_abse");
		abse /= p->um->getDomainSize();

		// Check for convergence.
		if(abse < p->um->getErrorTolerance()) {p->convergence[p->d_ID] = true;}	

		if(p->d_ID == 0)
		{
			bool converged = true;
			for(int d = 0; d < p->d_cnt; d++)
			{
				converged = converged && p->convergence[d];
			}
			p->um->converged = converged;
		}
	}

	void urbMulti::swapDomainMulti(TGPUurbPlan* p) 
	{
		// Each device (but 0) handles its lower boundary.
		if(p->d_ID != 0)
		{
			int lbI = (p->d_ID - 1) * 2; 	// Index where lower bottom info should be.
			int ltI = lbI + 1;				// Index where lower top info should be.
			cudaMemcpy(p->d_swp_rcvrs[lbI], p->d_swp_sndrs[lbI], p->swp_szs[lbI] * sizeof(float), cudaMemcpyDeviceToDevice);
			showError("urbMulti::swapDomainMulti -- first memcpy");
			cudaMemcpy(p->d_swp_rcvrs[ltI], p->d_swp_sndrs[ltI], p->swp_szs[ltI] * sizeof(float), cudaMemcpyDeviceToDevice);
			showError("urbMulti::swapDomainMulti -- second memcpy");
		}
	}

	void urbMulti::spliceDomainMulti(TGPUurbPlan* p) 
	{
		// Transfer info from device 0 to other devices along block allocation splitting scheme.
		int lmnt_ffst = p->grp_ffst * p->threads;

		//cudaMemcpy(&p->d_p1[lmnt_ffst], &p->d_p1[lmnt_ffst], p->grp_cnt * p->threads * sizeof(float), cudaMemcpyDeviceToDevice);
		showError("urbMulti::spliceDomainMulti -- memcpy p->d_p1[lmnt_ffst]");
		//cudaMemcpy(&p->d_p2[lmnt_ffst], &p->d_p2[lmnt_ffst], p->grp_cnt * p->threads * sizeof(float), cudaMemcpyDeviceToDevice);
		showError("urbMulti::spliceDomainMulti -- memcpy p->d_p2[lmnt_ffst]");
	}
}
