#include "urbSetup.h"

namespace QUIC
{
	void urbSetup::setup(QUIC::urbModule* um)
	{
		if(um->use_fortran)
		{
			QUIC::urbSetup::usingFortran(um);
		}
		else
		{
			QUIC::urbSetup::usingCPP(um);
		}
	}

	void urbSetup::usingFortran(QUIC::urbModule* um)
	{
		std::string directory = um->input_directory;
	
		if(directory.at(directory.length() - 1) != '/') 
		{
			directory = directory.append("/");
		}
	
		um->qwrite("###--- Fortran setup ---###\n");
	
		char* orig_dir = getcwd(NULL, 0);
		chdir_or_die(directory);
	
		um->stpwtchs->parse->start();
		init_();
		um->sim_params_parsed = true;
		um->stpwtchs->parse->stop();
		um->qwrite("  ...init_() done.\n");

		// Now we can to push the info to the urbModule.
		
		// First go the basic parameters
		um->nx = (unsigned) F90DATAMODULE_(nx) - 1; um->gx = um->nx + 1;
		um->ny = (unsigned) F90DATAMODULE_(ny) - 1; um->gy = um->ny + 1;
		um->nz = (unsigned) F90DATAMODULE_(nz) - 1; um->gz = um->nz + 1;

		um->dx = F90DATAMODULE_(dx);
		um->dy = F90DATAMODULE_(dy);
		um->dz = F90DATAMODULE_(dz);
		
		um->A      = F90DATAMODULE_(a);
		um->B      = F90DATAMODULE_(b);
		um->alpha1 = F90DATAMODULE_(alpha1);
		um->alpha2 = F90DATAMODULE_(alpha2);

		um->residual_reduction = F90DATAMODULE_(residual_reduction);

		// Now the datamodule must be initialized.
		//um->initialize(); Now called from urbParser
		
		um->stpwtchs->sort->start();
		sort_(); 
		um->stpwtchs->sort->stop();
		um->qwrite("  ...sort_() done.\n");

		F90DATAMODULE_(i_time) = 1; 
		um->qwrite("  ...doing just one time step.\n");

		um->stpwtchs->sensor->start();
		sensorinit_();	
		um->stpwtchs->sensor->stop();
		um->qwrite("  ...sensorinit_() done.\n");
			
		um->stpwtchs->bldngprm->start();
		building_parameterizations_();
		um->stpwtchs->bldngprm->stop();
		um->qwrite("  ...building_params_() done.\n");
	
		um->stpwtchs->denoms->start();
		denominators_();
		um->stpwtchs->denoms->stop();
	
		chdir_or_die(orig_dir);
		free(orig_dir);
		
		// Finally the information Fortran produced can be transferred to the module.				
		QUIC::urbSetup::compressBoundaries
		(
			um, 
			F90DATAMODULE_(e), F90DATAMODULE_(f), 
			F90DATAMODULE_(g), F90DATAMODULE_(h), 
			F90DATAMODULE_(m), F90DATAMODULE_(n), 
			F90DATAMODULE_(o), F90DATAMODULE_(p), F90DATAMODULE_(q)
		);
		
		for(unsigned int i = 0; i < um->nx*um->ny*um->nz; i++)
		{
			um->h_typs.c[i] = (CellType) F90DATAMODULE_(icellflag)[i];
		}
		
		copyDoubleToFloat(um->h_ntls.u, F90DATAMODULE_(uo), um->gx, um->gy, um->gz);
		copyDoubleToFloat(um->h_ntls.v, F90DATAMODULE_(vo), um->gx, um->gy, um->gz);
		copyDoubleToFloat(um->h_ntls.w, F90DATAMODULE_(wo), um->gx, um->gy, um->gz);
		
		um->transferDataToDevice();
		um->reset();
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
		//sortBuildings(um->buildings);
		um->stpwtchs->sort->stop();
		
		QUIC::urbSetup::buildingParameterizations(um);
		QUIC::urbSetup::streetIntersections(um); // if applicable

		um->transferDataToDevice();
		
		QUIC::urbSetup::setupBoundaryMatrices(um); // CUDA-nized.
		
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
			um->sensors[i]->determineVerticalProfiles(um->buildings.zo, um->dz, 0);
		}
		
		um->qwrite("done.\n");
	}
	
	void urbSetup::determineInitialVelocities(QUIC::urbModule* um)
	{	
		if(um->sensors.size() == 1)
		{
			if(um->sensors[0]->prfl_lgth != (int) um->h_ntls.dim.z)
			{
				std::cerr << "Error in " << __FILE__ << " : " << __func__ << std::endl;
				std::cerr << "Sensor profile length and velocities z dimension are wrong." << std::endl;
				std::cerr << "  profile length = " << um->sensors[0]->prfl_lgth << std::endl;
				std::cerr << "  h_ntls.dim.z   = " << um->h_ntls.dim.z << std::endl;
				return; 
			}
			
			um->qwrite("Determining velocities...");
			
			//std::cout << "sensors.size() = " << um->sensors.size() << std::endl;
			//std::cout << "sensors[0]->prfl_lgth = " << um->sensors[0]->prfl_lgth << std::endl;
			//std::cout << "h_ntls.dim.y = " << um->h_ntls.dim.y << std::flush;
			//std::cout << " h_ntls.dim.x = " << um->h_ntls.dim.x << std::endl;
			
			for(int k = 1; k < um->sensors[0]->prfl_lgth; k++)
			for(unsigned j = 0; j < um->h_ntls.dim.y; j++)
			for(unsigned i = 0; i < um->h_ntls.dim.x; i++)
			{
				int ndx = k*um->h_ntls.dim.y*um->h_ntls.dim.x + j*um->h_ntls.dim.x + i;
				um->h_ntls.u[ndx] = um->sensors[0]->u_prof[k];
				um->h_ntls.v[ndx] = um->sensors[0]->v_prof[k];
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
		
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->initialize(um->h_ntls, um->dx, um->dy, um->dz);
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
		
		float dx = um->dx;
		float dy = um->dy;
		float dz = um->dz;
		
		// Fill the celltypes matrix with "fluid" cells.
		for(unsigned int m = 0; m < um->domain_size; m++) {um->h_typs.c[m] = FLUID;}
	
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->interior(um->h_typs, um->h_ntls, dx, dy, dz);
		}
//	std::cout << "interior done..." << std::flush;
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->upwind(um->h_typs, um->h_ntls, dx, dy, dz);
		}
//	std::cout << "upwind done..." << std::flush;		
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->wake(um->h_typs, um->h_ntls, dx, dy, dz);
		}
//	std::cout << "wake done..." << std::flush;
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->canyon(um->h_typs, um->h_ntls, um->urbBuildings, dx, dy, dz);
		}
//	std::cout << "canyon done..." << std::flush;		
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->rooftop(um->h_typs, um->h_ntls, dx, dy, dz);
		}
//	std::cout << "rooftop done..." << std::flush;		
		for(unsigned int i = 0; i < um->urbBuildings.size(); i++)
		{
			um->urbBuildings[i]->interior(um->h_typs, um->h_ntls, dx, dy, dz);
		}
//	std::cout << "interior done..." << std::flush;		
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
			QUIC::intersect::poisson(um->h_ntls, um->h_bndrs, um->h_typs, um->dx, um->dy, um->dz);

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
	    for(unsigned cI = 0; cI < um->domain_size; cI++)
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
		um->qwrite("Compressing boundaries (fortran only)...");
		um->stpwtchs->bndrymat->start();

		int nx = um->nx;
		int ny = um->ny;
		int nz = um->nz;
		
		int row = nx;
		int slc = nx*ny;

		//for(unsigned int i = 0; i < um->domain_size; i++) 		
		for(int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
		for(int i = 0; i < nx; i++)
		{
			int cI = k*slc + j*row + i;
		
			QUIC::encodeBoundary
			(
				um->h_bndrs.cmprssd[cI],
				(float) e[cI], (float) f[cI],
				(float) g[cI], (float) h[cI],
				(float) m[cI], (float) n[cI],
				(float) o[cI], (float) p[cI], (float) q[cI]
			);
			
			QUIC::encodePassMask(um->h_bndrs.cmprssd[cI], (i + j + k) & 1);
			
			bool slcBndry = (k == 0 || k == nz - 1);
			bool rowBndry = (j == 0 || j == ny - 1);
			bool colBndry = (i == 0 || i == nx - 1);
			
			QUIC::encodeDomainBoundaryMask(um->h_bndrs.cmprssd[cI], slcBndry, rowBndry, colBndry);
		}
		
		um->stpwtchs->bndrymat->stop();
		um->qwrite("done.\n");
	}
}

