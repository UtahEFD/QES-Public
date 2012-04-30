/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Investigation into using multiple GPUs.
*/

#ifndef URBMULTI_H
#define URBMULTI_H

#include "urbModule.h"
#include "urbCUDA.h"
#include "multithreading.h"

// From Atul, who got it from Quinn (OpenMPI book)
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)

extern "C" void showError(char const*);

namespace QUIC
{

	typedef struct
	{
		int d_ID;
		int d_cnt;
		int threads;	// elements per block
		bool* convergence;
		float** d_swp_sndrs;
		float** d_swp_rcvrs;
		int* swp_szs;
		int grp_ffst;
		int grp_cnt;

		QUIC::urbModule* um;

		float *d_abse, *d_err;
	
		float *d_e, *d_f, *d_g, *d_h, *d_m, *d_n, *d_r;
		float *d_p1, *d_p2;

	} TGPUurbPlan;

	class urbMulti : private urbCUDA
	{
		private:
			static void iterateMulti(TGPUurbPlan*, int);
			static void checkForConvergenceMulti(TGPUurbPlan*);
			static void swapDomainMulti(TGPUurbPlan*);

			static void pallocMulti(TGPUurbPlan*);
			static void setSwapPointersAndSizes(TGPUurbPlan*);
			static void splitDomainMulti(TGPUurbPlan*);
			static void spliceDomainMulti(TGPUurbPlan*);

			static CUT_THREADPROC threadedSORSolver(TGPUurbPlan*);

		public:
			/**
			* Solves the problem using multiple GPUs and the SOR Red-Black method.
			*
			* @param omegarelax an important parameter for the SOR method. Default: 1.78.
			*/
			static void solveUsingMultiGPU(QUIC::urbModule&, float);
			static void solveUsingMultiGPU(QUIC::urbModule*, float);
	};
}

#endif

