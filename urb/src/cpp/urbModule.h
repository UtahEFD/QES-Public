/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb
* Source: Adapted from datamodule.f90 and other QUICurbv5.? Fortran files.
*/

#ifndef URBMODULE_H
#define URBMODULE_H

#include <vector>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "util/angle.h"
#include "util/Timer.h"
#include "quicutil/QUSensor.h"
#include "quicutil/QUMetParams.h"
#include "quicutil/QUSimparams.h"
#include "quicutil/QUFileOptions.h"
#include "quicutil/QUBuildings.h"
#include "quicutil/velocities.h"

#include "../util/matrixIO.h"

#include "boundaryMatrices.h"
#include "buildingList.h"
#include "celltypes.h"
#include "urbStopWatchList.h"

extern "C" void cudaZero(float* d_abse, size_t size, float value);
extern "C" void showError(char const*);
extern "C" void cudaCompDenoms(QUIC::boundaryMatrices d_bndrs, float omegarelax, float A, float B);

namespace QUIC 
{
	/**
	* Part of the QUIC namespace, the urbModule class keeps track of the current
	* state of a solution, holding device pionters and parameters. 
	* CUDA is used to solve in the urbCUDA class.
	*/
	
	static const float        DFLT_OMEGARELAX     =  1.78;
	static const int DFLT_MAX_ITERATIONS = 10000;
	
	class urbModule 
	{		
		friend class plumeModule; // Really probably shouldn't be friends.
		
		friend class urbParser;
		friend class urbSetup;
		
		friend class urbHost;
		friend class urbCUDA;
			
		public:

			urbModule();
			virtual ~urbModule();

			void initialize();
			void reset();

			/**
			* After this call the user is responsible for freeing the allocated memory.
			* The allocated block will be of size nx * ny * nz * sizeof(float).
			*
			* @returns if the solution has converged, then a host pointer to dynamically 
			* allocated memory will be returned; otherwise, NULL is returned.
			*/
			float* getSolution() const;

			int getDomainSize() const;
			int getGridSize() const;
			
			int getNX() const;
			int getNY() const;
			int getNZ() const;
			
			int getGX() const;
			int getGY() const;
			int getGZ() const;
			
			float getDX() const;
			float getDY() const;
			float getDZ() const;

			std::string getName() const;
			void setName(std::string const& newName);

			bool isConvergedQ() const;

			/** Default: 1.0 until at least one iteration executed. @returns the calculated error tolerance. */
			float getErrorTolerance() const;
			/** Default: 1.0 until at least one iteration executed. @returns the calculated error tolerance. */
			float getEpsilon() const;
			/** Sets runto_eps so that it is used rather than the residual reduction by default. */
			void setErrorTolerance(float const& _runto_eps);
			/** Sets runto_eps so that it is used rather than the residual reduction by default. */
			void setEpsilon(float const& _runto_eps);
			/** Gets the error of the last iteration. */
			float getError() const;

			/** 
			* Sets the value of the omegarelax parameter. Default: 1.78. 
			* Limited to [0.0, 2.0].
			*/
			void setOmegaRelax(float const& _omegarelax = 1.78f);
			float getOmegaRelax() const;
			
			void setWindAngle(sivelab::angle const& a);

			int getMaxIterations() const;
			void setMaxIterations(int const&);
			int getIteration() const;
			int getIterationStep() const;
			void setIterationStep(int const&);


			int getDiffusionStep() const;
			bool isDiffusionOnQ() const;
			void turnDiffusionOn();
			void turnDiffusionOff();

      bool isHostRunnableQ() const;

			float getTotalTime() const;

			bool isQuietQ() const;
			void beQuiet(bool const& q = true);

			void printLastError() const;
			
			/**
			* This method should run all methods that validate information, returning
			* false is can check fails, along with some sort of output, hopefully...
			*/
			bool sanityCheck() const;
			
		public:
			// Other File Options
			bool output_celltypes;
			bool output_boundaries;
			bool output_divergence;
			bool output_denominators;
			bool output_lagrangians;
			bool output_velocities;
			bool output_viscocity;
			// Other File Options

			// Default: false. True => many things output to the console.
			bool quiet;
			bool use_fortran; // Whether to use fortran or not for setup.
			bool host_runnable;
			
		protected:
		
			std::string name;
			std::string input_directory;

			// Domain Information
			float A; 
			float B; 
			float alpha1; 
			float alpha2; 
			float eta;
			
			int gx; 
			int gy; 
			int gz;

      int slice_size;
			int domain_size;
			int grid_size;
			// Domain Information

			// Sim Parameters
      bool sim_params_parsed;

            quSimParams simParams;
            quMetParams metParams;
            
            std::vector<quSensorParams> sensors;

		  quBuildings fileBuildings;
		  urbBuildingList buildings;

      quFileOptions fileOptions;
      
			// Iteration Parameters
			bool converged;

			float omegarelax; 
			float one_less_omegarelax;
						
			int iteration;
			int iter_step;
			
			float eps;
			float runto_eps;
			float abse;
			// Iteration Parameters

			// Diffusion Parameters
			int diffusion_iter;
			// Diffusion Parameters

			// Unclassified
		  float Lx;
		  float Ly;
		  float Lz;
		  // Unclassified

      bool module_initialized;

			// Host data structures
			boundaryMatrices h_bndrs; // cmprssd
			celltypes  h_typs; // celltypes (icellflags)
			velocities h_ntls; // initial velocities (uo, vo, wo)
			velocities h_vels; // velocities(u, v, w)
			
			// Device data structures
			boundaryMatrices d_bndrs; // cmprssd and denoms are device pointers.
			celltypes  d_typs; // celltypes (icellflags)
			velocities d_vels; // velocities(u, v, w)

			// urbHost Pointers only.
			float* h_r;
			float* h_p1;
			float* h_p2_err;
			// Host Pointers
			
			// Device Pointers
			float* d_r;
			float* d_p1;
			float* d_p2_err; // Stores old iteration and used for finding iteration error.
			float* d_abse;
			float* d_visc;
			// Device Pointers

      // Timing
	    urbStopWatchList* stpwtchs;

    protected:			
			void setDimensions(int nx, int ny, int nz);
			
			bool validBuildingsQ() const;
			bool validDimensionsQ(std::string checking_loc) const;

      // Quiet write. Only writes if module->quiet = false;
			void qwrite(std::string const& message) const;

      // Returns a pointer to newly allocated host memory that contains the
			// data found in d_ptr.
			template <typename T>
      static inline T* getCUDAdata
      (
        const T* d_ptr, 
        int const& x, int const& y, int const& z
      )
      {
        if(d_ptr == NULL) {return NULL;}
	      else 
	      {
		      int size = x*y*z;
		      T* temp = new T[size];
		      cudaMemcpy(temp, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
		      return temp;
	      }
      }			
			
			// TODO Need to add option for Device or Host memory pointers.  And make it public.
			velocities getFinalVelocities() const;
			celltypes getCellTypes() const;
			
		private:
			
			/**
			* Checks the values of the input parameters for the SOR scheme.
			* A must be positive.
			* B must be positive.
			* dx must be positive.	
			* nx must be greater than 1, a power of 2 and a multiple of 64. 
			* ny must be greater than 1 and a power of 2.
			* nz must be greater than 2.
			* 
			* @returns true if parameters are correctly specified.
			*/
			bool validParametersQ() const;

      /**
      * Checks the values inside the boundary matrices for acceptable values.
      * Coming out of a move form 32 to 64-bit and the boundary matrices are 
      * negative.
      *
      * A boundary matrix entry must be positive between 1 and 2^?
      */
      bool validBoundaryMaskQ() const;			
			
			/**
      * Checks the values of the denominators for SOR calculation for valid values.
      * This doesn't mean that they are correct, just one of the options possible.
      */
		  bool validDenominatorsOnDeviceQ() const;
			  
			/**
			* Ensures that the device has enough memory before transferring the data.
			* Returns positive amount of memory needed if enough.
								Otherwise the deficit, which is negative.
			*/
			float enoughDeviceMemoryQ() const;		
			
			
			/**
			* Allocates the needed memory on the device for all the matrices.
			* Calls enoughDeviceMemoryQ to ensure there is enough first.
			*/
			void allocateDeviceMemory();
			void deallocateDeviceMemory();
			
			void allocateHostMemory();
			void deallocateHostMemory();
			
			void allocateHostRunnableMemory();
			void deallocateHostRunnableMemory();
			/**
			* Transfers the boundary matrices, celltypes and initial velocity matrices
			* to the device in preperation for SOR scheme.
			*/
			void transferDataToDevice();
			
			
			/**
			* Checks the device pointers and the corresponding memory sizes.
			*
			* @returns true  - if all device pointers are not NULL and sizes check-out.
			*					 false - otherwise.
			*/
			bool validDevicePointersQ(bool err = true) const;
			bool validHostPointersQ(bool err = true) const;
			
			// Returns true if any data if flagged for output.
			bool anyOutputQ() const;
			
			static bool badPointerQ
			(
				const void* ptr, 
				std::string const& name, 
				std::string const& loc,
				bool err
			);
			
			
			/**
			* Verifies that the given pointers are NOT NULL.
			*
			* @returns false if any one pointer is NULL.
			*/
			static bool validPointersQ
			(
				boundaryMatrices const& bndry_mats, 
				celltypes const& p_celltypes,
				velocities const& vels,
				std::string const& loc = "host",
				bool err = true
			);
	};
}

#endif

