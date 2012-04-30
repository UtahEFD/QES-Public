/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Providing HOST side RB implementation for comparison purposes.
*/

#ifndef URBHOST_H
#define URBHOST_H

#include <iostream>
#include <string>
#include <cstring>

#include "urbModule.h"

namespace QUIC 
{

	/**
	* Part of the QUIC namespace, urbHOST is a stateless class that runs QUIC uses
	* the host processor on an urbModule. Uses C++ to solve the SOR 
	* iteration technique found in sor3d.f90.
	* 
	* urbCUDA and urbHost use the same interface that should be abstracted out for
	* future use by other modules that want to solve the same problem.
	*/
	class urbHost 
	{
		
		friend class urbViewer;

		protected:

			/** Executes the first iteration. **/
			static bool firstIteration(QUIC::urbModule*);
			
			/**
			* Executes a specified number of iterations without checking for 
			* convergence.
			*
			* @param times is the number of iterations to execute. Default: 1.
			*/
			static void iterate(QUIC::urbModule*, int const& times);
			
			/**
			* Compares the current iterate with what's stored as the old iterate. 
			* The distance between the old iterate and the new iterate is not always 
			* one.
			*
			* #NOTE: See iterate.
			*/
			static void checkForConvergence(QUIC::urbModule*);
			
			/**
			* Calls euler kernel, using the current lagragian field to calculate 
			* u, v and w.
			*/
			static void calcVelocities(QUIC::urbModule*);


		public:
				
			/**
			* Solves the problem using solveUsingSOR_RB with omegarelax set to 1.0.
			*/
			static void solveUsingGS_RB(QUIC::urbModule*);
			
			/**
			* Solves the problem using a successive overrelaxation red black technique.
			* The red black portion of the technique is required for parallel 
			* implementation and is included here for host calculation comparison.
			*
			* @param omegarelax an important parameter for the SOR method. Default: 1.78.
			*/
			static void solveUsingSOR_RB(QUIC::urbModule*);
			
			/**
			* Solves the problem using diffusion after each solution is found using 
			* Red-Black SOR technique and euler is applied. Then sor3d and euler are 
			* run again and diffusion is applied. This is done <emph>steps</emph>
			* times. Diffusion modifies initial velocity matrices uo, vo and wo.
			*
			* @param step number of times to call diffusion and refind solution.
			*/
			static void solveWithDiffusion(QUIC::urbModule*, int const& steps);
	};
}

#endif
