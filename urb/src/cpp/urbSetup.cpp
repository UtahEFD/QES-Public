#include "urbSetup.h"

#include "../util/directory.h"

#include "intersect.h"

namespace QUIC
{
	void urbSetup::setup(QUIC::urbModule* um)
	{
            QUIC::urbSetup::usingCPP(um);
	}

	void urbSetup::usingCPP(QUIC::urbModule* um)
	{
		um->qwrite("###--- CPP setup ---###\n");
		um->stpwtchs->sensor->start();
			
		QUIC::urbSetup::initializeSensors(um);
		QUIC::urbSetup::determineInitialVelocities(um);
		
		um->stpwtchs->sensor->stop();

		QUIC::urbSetup::initializeBuildings(um);

		um->stpwtchs->sort->start();
		um->buildings.sort();
		um->stpwtchs->sort->stop();
		
		QUIC::urbSetup::buildingParameterizations(um);
		QUIC::urbSetup::streetIntersections(um); // if applicable

		um->transferDataToDevice();
		
		QUIC::urbSetup::setupBoundaryMatrices(um); // CUDA-nized.

    std::cout << um->simParams.nx << "." << um->simParams.ny << "." << um->simParams.nz <<std::endl;
		
		um->qwrite("###--- done ---###\n\n");
		um->reset();
	}
	
	void urbSetup::usingCUDA(QUIC::urbModule* um)
	{
		std::cout << "urbSetup::usingCUDA(~~) is not implemented yet." << std::endl;
	}

	void urbSetup::initializeSensors(QUIC::urbModule* um)
	{
		um->qwrite("Initializing sensors...");
		
		for(unsigned int i = 0; i < um->sensors.size(); i++)
		{
			um->sensors[i].determineVerticalProfiles(um->buildings[0].zo, um->simParams.dz, 0);
		}
		
		um->qwrite("done.\n");
	}
	
	void urbSetup::determineInitialVelocities(QUIC::urbModule* um)
	{	
		if(um->sensors.size() == 1)
		{
			if(um->sensors[0].prfl_lgth != (int) um->h_ntls.dim.z)
			{
				std::cerr << "Error in " << __FILE__ << " : " << __func__ << std::endl;
				std::cerr << "Sensor profile length and velocities z dimension are wrong." << std::endl;
				std::cerr << "  profile length = " << um->sensors[0].prfl_lgth << std::endl;
				std::cerr << "  h_ntls.dim.z   = " << um->h_ntls.dim.z << std::endl;
				return; 
			}
			
			um->qwrite("Determining velocities...");
			
			//std::cout << "sensors.size() = " << um->sensors.size() << std::endl;
			//std::cout << "sensors[0].prfl_lgth = " << um->sensors[0].prfl_lgth << std::endl;
			//std::cout << "h_ntls.dim.y = " << um->h_ntls.dim.y << std::flush;
			//std::cout << " h_ntls.dim.x = " << um->h_ntls.dim.x << std::endl;
			
			for(int k = 1; k < um->sensors[0].prfl_lgth; k++)
			for(int j = 0; j < um->h_ntls.dim.y; j++)
			for(int i = 0; i < um->h_ntls.dim.x; i++)
			{
				int ndx = k*um->h_ntls.dim.y*um->h_ntls.dim.x + j*um->h_ntls.dim.x + i;
				um->h_ntls.u[ndx] = um->sensors[0].u_prof[k];
				um->h_ntls.v[ndx] = um->sensors[0].v_prof[k];
			}
			
			um->qwrite("done.\n");
		}
		else // Barnes mapping scheme.
		{
			std::cerr 
			<< 
				"Unable to average sensor distances for multiple sensors." 
			<< 
			std::endl;
		}
	}
	
	void urbSetup::initializeBuildings(QUIC::urbModule* um)
	{
		um->qwrite("Initializing buildings...");
		um->stpwtchs->init->start();
		
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].initialize(um->h_ntls, um->simParams.dx, um->simParams.dy, um->simParams.dz);
		}
		
		QUIC::urbSetup::checkForVegetation(um);
		
		um->stpwtchs->init->stop();
		
		um->qwrite("done.\n");
	}
	
	
	
	void urbSetup::checkForVegetation(QUIC::urbModule* um)
	{
		
	}
	
	
	
	void urbSetup::buildingParameterizations(QUIC::urbModule* um)
	{
		um->qwrite("Parameterizing buildings...");
		um->stpwtchs->bldngprm->start();
		
		if(!um->validDimensionsQ("urbSetup::buildingParam()")) {return;}
		
		float dx = um->simParams.dx;
		float dy = um->simParams.dy;
		float dz = um->simParams.dz;
		
		// Fill the celltypes matrix with "fluid" cells.
		for(int m = 0; m < um->domain_size; m++) 
		{
		  um->h_typs.c[m] = FLUID;
		}
	
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].interior(um->h_typs, um->h_ntls, dx, dy, dz);
		}
	std::cout << "interior done..." << std::flush;
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].upwind(um->h_typs, um->h_ntls, dx, dy, dz);
		}
	std::cout << "upwind done..." << std::flush;		
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].wake(um->h_typs, um->h_ntls, dx, dy, dz);
		}
	std::cout << "wake done..." << std::flush;
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].canyon(um->h_typs, um->h_ntls, um->buildings, dx, dy, dz);
		}
	std::cout << "canyon done..." << std::flush;		
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].rooftop(um->h_typs, um->h_ntls, dx, dy, dz);
		}
	std::cout << "rooftop done..." << std::flush;		
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			um->buildings[i].interior(um->h_typs, um->h_ntls, dx, dy, dz);
		}
	std::cout << "interior done..." << std::flush;		
		// Zero the bottom slice. Necessary? Yes!! Do the velocities too!!
		int nx = um->h_typs.dim.x;
		int ny = um->h_typs.dim.y;

		int gx = um->gx;
		int gy = um->gy;
		
		int grid_slc = gx*gy;
		
		for(int j = 0; j < ny; j++)
		for(int i = 0; i < nx; i++)
		{
			int cellndx = j*nx + i;
		  
			um->h_typs.c[cellndx] = SOLID;
			
			int gI   = j*gx + i;
		  int gI_k = gI + grid_slc;
		  
			um->h_ntls.u[gI]   = um->h_ntls.v[gI] = um->h_ntls.w[gI] = 0.;
			um->h_ntls.w[gI_k] = 0.;
		}
		
		um->stpwtchs->bldngprm->stop();
		um->qwrite("done.\n");
	}
	
	void urbSetup::streetIntersections(QUIC::urbModule* um)
	{
		if(um->simParams.intersection_flag)
		{
			um->qwrite("Street Intersections...");
			um->stpwtchs->intersect->start();
			
			// Do the street intersections.
			QUIC::intersect::street (um->h_typs);
			QUIC::intersect::poisson(um->h_ntls, um->h_bndrs, um->h_typs, um->simParams.dx, um->simParams.dy, um->simParams.dz);

			um->stpwtchs->intersect->stop();
			um->qwrite("done.\n");
		}
	}
		
	void urbSetup::setupBoundaryMatrices(QUIC::urbModule* um)
	{
		um->qwrite("Setting up boundary matrices (including compression)...");
		um->stpwtchs->bndrymat->start();
		
		// Recently integrated. May present issues / cause problems. \\todo verify
		// Setup boundary matrices in d_bndrs.cmprssd
		cudaSetupBndryMats(um->d_bndrs, um->d_typs);
		cudaThreadSynchronize();
	
	  if(um->host_runnable)
	  {
	    // Setup the compressed boundary mats in host memory...
	    // This addition is getting messy. Need to find a better way...
	    for(int cI = 0; cI < um->domain_size; cI++)
	    {
	      QUIC::determineBoundaryCell(um->h_bndrs.cmprssd[cI], um->h_typs, cI);
	    }
	    
	    std::cout << "h_bndrs.cmprssd finished." << std::endl;
	  }  
	  
		um->stpwtchs->bndrymat->stop();
		//um->qwrite("done.\n");
	}
	
	void urbSetup::compressBoundaries
	(
		QUIC::urbModule* um,
		double* e, double* f, 
		double* g, double* h, 
		double* m, double* n,
		double* o, double* p, double* q
	)
	{
		um->qwrite("Compressing boundaries...");
		um->stpwtchs->bndrymat->start();

		//for(unsigned int i = 0; i < um->domain_size; i++) 		

		int row = um->simParams.nx;
		int slc = um->simParams.nx*um->simParams.ny;
		for(int k = 0; k < um->simParams.nz; k++)
		for(int j = 0; j < um->simParams.ny; j++)
		for(int i = 0; i < um->simParams.nx; i++)
		{
			int cI = k*slc + j*row + i;

      um->h_bndrs.cmprssd[cI] = 0;		
			QUIC::encodeBoundary
			(
				um->h_bndrs.cmprssd[cI],
				(float) e[cI], (float) f[cI],
				(float) g[cI], (float) h[cI],
				(float) m[cI], (float) n[cI],
				(float) o[cI], (float) p[cI], (float) q[cI]
			);
			
			QUIC::encodePassMask(um->h_bndrs.cmprssd[cI], (i + j + k) & 1);
			
			bool slcBndry = (k == 0 || k == um->simParams.nz - 1);
			bool rowBndry = (j == 0 || j == um->simParams.ny - 1);
			bool colBndry = (i == 0 || i == um->simParams.nx - 1);
			
			QUIC::encodeDomainBoundaryMask(um->h_bndrs.cmprssd[cI], slcBndry, rowBndry, colBndry);
		}
		
		um->stpwtchs->bndrymat->stop();
		um->qwrite("done.\n");
	}
}

