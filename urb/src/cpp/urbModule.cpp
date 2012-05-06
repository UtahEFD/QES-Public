#include "urbModule.h"

namespace QUIC 
{

	urbModule::urbModule()
	{
		name            = "";
		input_directory = ".";
				
		quiet = use_fortran = false;
		
		output_celltypes   = output_boundaries   = false; 
		output_divergence  = output_denominators = false;
		output_lagrangians = output_velocities   = false;
		output_viscocity   = false;
		
		// Domain Parameters and Pointers
		// Device and Host Pointers
		this->setDimensions(0, 0, 0);
		
		h_bndrs.cmprssd = d_bndrs.cmprssd = NULL;
		h_bndrs.denoms  = d_bndrs.denoms  = NULL;

		h_typs.c = d_typs.c = NULL;

		h_ntls.u = d_vels.u = NULL;
		h_ntls.v = d_vels.v = NULL;
		h_ntls.w = d_vels.w = NULL;
		// Device and Host Pointers

    // Host pointers for urbHost ONLY
    h_r = h_p1 = h_p2_err = NULL;
    host_runnable = true;

		// Device only pointers
		d_r = NULL;
		
		d_p1   = d_p2_err = NULL;
		d_abse = NULL;
		
		d_visc = NULL;
		// Device only Pointers
		// Domain Parameters

		// Met Parameters
		sensors.resize(0);
	  // Met Parameters

		// Iteration Parameters
		converged = false;

		one_less_omegarelax = 1. - (omegarelax = DFLT_OMEGARELAX);
					
		iteration = 0;
		iter_step = 1;
		simParams.max_iterations = DFLT_MAX_ITERATIONS;
		
		eps = 1.;
		runto_eps = abse = simParams.residual_reduction = 0.;
		// Iteration Parameters


		// Domain Information
    alpha1 = alpha2 = 1.;
		eta = (alpha1 * alpha1) / (alpha2 / alpha2);

		A = B = 0.; // A = (dx * dx) / (dy * dy);	B = eta * (dx * dx) / (dz * dz);
		// Domain Information
		

		// Diffusion Parameters
		simParams.diffusion_flag = false;
		simParams.diffusion_step = 0;
		diffusion_iter = 1;
		// Diffusion Parameters
		

		// Unclassified
	  Lx = Ly = Lz = 0.;
	  // Unclassified

	  //Timings
    stpwtchs = new urbStopWatchList();
		
		sim_params_parsed = false;
	}

	urbModule::~urbModule() 
	{
		for(unsigned i = 0; i < sensors.size(); i++)
		{
			delete sensors[i];
		}
		
		this->deallocateHostMemory();
		this->deallocateDeviceMemory();
		
		delete stpwtchs;
	}

	void urbModule::initialize()
	{
		if(!sim_params_parsed)
		{
			std::cerr << "Unable to initialize urbModule." << std::endl;
			std::cerr << "sim_params not parsed correctly." << std::endl;
			return;
		}
	
	    stpwtchs->init->start();
	
		// Simparams // SOR
		if(simParams.quic_cfd_type > 0) 
		{
			simParams.max_iterations = 1; 
			simParams.diffusion_flag = false;
		}
		
		this->setDimensions(simParams.nx, simParams.ny, simParams.nz);
		
		A = (simParams.dx * simParams.dx) / (simParams.dy * simParams.dy);
		B = eta * (simParams.dx * simParams.dx) / (simParams.dz * simParams.dz);

		
		// Unclassified
		Lx = simParams.nx * simParams.dx;
		Ly = simParams.ny * simParams.dy;
		Lz = simParams.nz * simParams.dz;		

						
		// Buildings
		// todo resolve float to int warning.
		fileBuildings.x_subdomain_sw *= simParams.dx;
		fileBuildings.y_subdomain_sw *= simParams.dy;
		fileBuildings.x_subdomain_ne *= simParams.dx;
		fileBuildings.y_subdomain_ne *= simParams.dy;
		
		  stpwtchs->init->stop();
			
		// Allocate Matrices.
		this->allocateDeviceMemory();
		this->allocateHostMemory();
		this->allocateHostRunnableMemory();
		
		module_initialized = true;
		
		this->reset();
	}

	void urbModule::reset() 
	{
		converged = false;
		iteration = 0;
		eps 	  = 1.;
		abse 	  = 0.;

		simParams.diffusion_step = 0;

    // Check method...
    stpwtchs->resetIterationTimers();

		cudaZero(d_p1,     domain_size, 0.);
		cudaZero(d_p2_err, domain_size, 0.);
		
		if(host_runnable)
		{
		  for(int i = 0; i < domain_size; i++)
		  {
		    h_p1[i] = h_p2_err[i] = 0.;
		  }
		}
		
		int grid_data_size = grid_size*sizeof(float);
		std::string prefix = "urbModule::reset() - ";
		
		cudaMemcpy(d_vels.u, h_ntls.u, grid_data_size, cudaMemcpyHostToDevice);	
			showError((prefix + "d_u").c_str());
		cudaMemcpy(d_vels.v, h_ntls.v, grid_data_size, cudaMemcpyHostToDevice);	
			showError((prefix + "d_v").c_str());
		cudaMemcpy(d_vels.w, h_ntls.w, grid_data_size, cudaMemcpyHostToDevice);	
			showError((prefix + "d_w").c_str());
	}

	float* urbModule::getSolution() const
	{
		if(this->isConvergedQ()) 
		{
			float* sol_copy = new float[domain_size];
			cudaMemcpy(sol_copy, d_p1, domain_size* sizeof(float), cudaMemcpyDeviceToHost);
			return sol_copy;
		}
		else {return NULL;}
	}

	int urbModule::getDomainSize() const {return domain_size;}
	int urbModule::getGridSize() const 	{return grid_size;}
	
	int urbModule::getNX() const {return simParams.nx;}
	int urbModule::getNY() const {return simParams.ny;}
	int urbModule::getNZ() const {return simParams.nz;}
	
	int urbModule::getGX() const {return gx;}
	int urbModule::getGY() const {return gy;}
	int urbModule::getGZ() const {return gz;}
	
	float urbModule::getDX() const {return simParams.dx;}
	float urbModule::getDY() const {return simParams.dy;}
	float urbModule::getDZ() const {return simParams.dz;}

	std::string urbModule::getName() const {return name;}
	void urbModule::setName(std::string const& newName) {name = newName;}
			
	bool urbModule::isConvergedQ() const	{return converged;}

	float urbModule::getErrorTolerance() const 
	{
		return (runto_eps > 0.) ? runto_eps : eps;
	}
	float urbModule::getEpsilon() const {return this->getErrorTolerance();}
	
	void urbModule::setErrorTolerance(float const& _runto_eps)
	{
		runto_eps = (_runto_eps > 0.) ? _runto_eps : 0. ;
	}
	void urbModule::setEpsilon(float const& _runto_eps)
	{
		this->setErrorTolerance(_runto_eps);
	}
	
	float urbModule::getError() const {return abse;}			

	float urbModule::getOmegaRelax() const {return omegarelax;}
	void urbModule::setOmegaRelax(float const& mgrlx /*= 1.78*/) 
	{		
		omegarelax = (0. < mgrlx && mgrlx < 2.) ? mgrlx           : 
																							DFLT_OMEGARELAX ;

		one_less_omegarelax = 1. - omegarelax;
	}

	void urbModule::setWindAngle(sivelab::angle const& theAngle)
	{
		if(module_initialized)
		{
			for(unsigned i = 0; i < sensors.size(); i++)
			{
				sensors[i]->direction = theAngle.radians(sivelab::ENG);
			}
		}
		else
		{
			std::cerr << "Error in urbModule::setWindAngle" << std::endl;
			std::cerr << "The wind angle cannot be set for an un-initialized urbModule." << std::endl;
		}
	}

	int urbModule::getMaxIterations() const {return simParams.max_iterations;}
	void urbModule::setMaxIterations(int const& new_max)
	{
		simParams.max_iterations = (new_max > 1) ? new_max : DFLT_MAX_ITERATIONS ;
	}

	int urbModule::getIteration() const 
	{
		return iteration;
	}

	int urbModule::getIterationStep() const {return iter_step;}
	void urbModule::setIterationStep(int const& new_step)
	{
		iter_step = (new_step > 1) ? new_step : 1 ;
	}

	int urbModule::getDiffusionStep() const {return simParams.diffusion_step;}
	
	bool urbModule::isDiffusionOnQ() const {return simParams.diffusion_flag;}
	void urbModule::turnDiffusionOn()  {simParams.diffusion_flag = true;}
	void urbModule::turnDiffusionOff() {simParams.diffusion_flag = false;}

  bool urbModule::isHostRunnableQ() const {return host_runnable;}

	float urbModule::getTotalTime() const
	{
		//return this->getSetupTime() + this->getIterationTime();
		//return stpwtchs->getTotalElapsedTime();
		// TODO!!
		return 0.f;
	}

	void urbModule::beQuiet(bool const& q) {quiet = q;}
	bool urbModule::isQuietQ() const {return quiet;}

	void urbModule::printLastError() const
	{
		std::cout << "CUDA : " << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}

  bool urbModule::sanityCheck() const
  {
    bool passed = true;
    passed &= validBuildingsQ();
		passed &= validDimensionsQ(__FILE__);
		passed &= validParametersQ();
		passed &= validBoundaryMaskQ();
		passed &= validDenominatorsOnDeviceQ();
		passed &= validDevicePointersQ();
		passed &= validHostPointersQ();
		
		std::cerr << "UrbModule " << ((passed) ? "passed" : "failed") << " sanity check." << std::endl;
		
		if (!passed) {exit(EXIT_FAILURE);}
    
    return passed;
  }

	void urbModule::setDimensions(int nx, int ny, int nz)
	{
		gx = nx + 1;
		gy = ny + 1;
		gz = nz + 1;
		
		slice_size  = nx*ny;
		domain_size = nx*ny*nz;
		grid_size 	= gx*gy*gz;
	
		h_bndrs.dim.x = d_bndrs.dim.x = nx;
		h_bndrs.dim.y = d_bndrs.dim.y = ny;
		h_bndrs.dim.z = d_bndrs.dim.z = nz;
		
		h_typs.dim.x = d_typs.dim.x = nx;
		h_typs.dim.y = d_typs.dim.y = ny;
		h_typs.dim.z = d_typs.dim.z = nz;

		h_ntls.dim.x = d_vels.dim.x = gx;
		h_ntls.dim.y = d_vels.dim.y = gy;
		h_ntls.dim.z = d_vels.dim.z = gz;
	}

  bool urbModule::validBuildingsQ() const
  {
    bool enoughBuildings = buildings.size() > 0;
    
    if (!enoughBuildings)
    {
      std::cerr << "Buildings are required for the simulation." << std::endl;
    }
    
    return enoughBuildings;
  }

	bool urbModule::validDimensionsQ(std::string checking_loc) const
	{
		if(simParams.nx < 1 || simParams.ny < 1 || simParams.nz < 1)
		{
			std::cerr << "Error in " << checking_loc << std::endl;
			std::cerr << "Dimensions are Bad." << std::endl;
			std::cerr << "urbModule cell Dim : " << simParams.nx << "x" << simParams.ny << "x" << simParams.nz << std::endl;

			return false;
		}

	  if(simParams.nx + 1 != gx || simParams.ny + 1 != gy || simParams.nz + 1 != gz)
		{
			std::cerr << "Error in " << checking_loc << std::endl;
			std::cerr << "Dimensions for cell domain and grid domain do not correspond correctly." << std::endl;
			std::cerr << "urbModule cell Dim : " << simParams.nx << "x" << simParams.ny << "x" << simParams.nz << std::endl;
			std::cerr << "urbModule grid Dim : " << gx << "x" << gy << "x" << gz << std::endl;		
			
			return false;
		}
	
		if
		(
			simParams.nx != (int) h_typs.dim.x ||
			simParams.ny != (int) h_typs.dim.y ||
			simParams.nz != (int) h_typs.dim.z ||
			
			simParams.nx != (int) d_typs.dim.x ||
			simParams.ny != (int) d_typs.dim.y ||
			simParams.nz != (int) d_typs.dim.z
		)
		{
		  std::cerr << "Error in " << checking_loc << std::endl;
			std::cerr << "Dimensions of celltypes don't match urbModule." << std::endl;
			std::cerr << "H-Celltypes Dim : " << h_typs.dim.x << "x" << h_typs.dim.y << "x" << h_typs.dim.z	<< std::endl;
			std::cerr << "D-Celltypes Dim : " << d_typs.dim.x << "x" << d_typs.dim.y << "x" << d_typs.dim.z	<< std::endl;
			std::cerr << "urbModule   Dim : " << simParams.nx << "x" << simParams.ny << "x" << simParams.nz << std::endl;
			
			return false;
		}
		
		if 
		(	
			gx != h_ntls.dim.x ||
			gy != h_ntls.dim.y ||
			gz != h_ntls.dim.z ||
			
			gx != d_vels.dim.x ||
			gy != d_vels.dim.y ||
			gz != d_vels.dim.z
		)
		{
			std::cerr << "Error in " << checking_loc << std::endl;
			std::cerr << "Dimensions of velocities don't match urbModule." << std::endl;
			std::cerr << "H-InitVels  Dim : " << h_ntls.dim.x << "x" << h_ntls.dim.y << "x" << h_ntls.dim.z << std::endl;
			std::cerr << "D-Velocity  Dim : " << d_vels.dim.x << "x" << d_vels.dim.y << "x" << d_vels.dim.z << std::endl;
			std::cerr << "urbModule   Dim : " << gx           << "x" << gy           << "x" << gz           << std::endl;
			
			return false;
		}

		return true;
	}

	void urbModule::qwrite(std::string const& message) const
	{
		if(!quiet) {std::cout << message << std::flush;}
	}

  velocities urbModule::getFinalVelocities() const
  {
    velocities vlcts;
    vlcts.u = QUIC::urbModule::getCUDAdata<float>(d_vels.u, gx, gy, gz);
    vlcts.v = QUIC::urbModule::getCUDAdata<float>(d_vels.v, gx, gy, gz);
    vlcts.w = QUIC::urbModule::getCUDAdata<float>(d_vels.w, gx, gy, gz);
    
    vlcts.dim.x = gx;
    vlcts.dim.y = gy;
    vlcts.dim.z = gz;
    
    return vlcts;
  }
  
	celltypes urbModule::getCellTypes() const
	{
	  celltypes clltyps;
	  clltyps.c = QUIC::urbModule::getCUDAdata<CellType>(d_typs.c, simParams.nx, simParams.ny, simParams.nz);
	  
	  clltyps.dim.x = simParams.nx;
	  clltyps.dim.y = simParams.ny;
	  clltyps.dim.z = simParams.nz;
	  
	  return clltyps;
	}
  

	bool urbModule::validParametersQ() const
	{
		bool parameters_good = true;

		if(A < 0.f) {std::cerr << "A < 0.0." << std::endl; parameters_good = false;}
		if(B < 0.f) {std::cerr << "A < 0.0." << std::endl; parameters_good = false;}

		//alpha1
		//alpha2		

		if(omegarelax < 0.f || 2.f < omegarelax) 
		{
			if(!quiet) {std::cerr << "omegarelax in [0., 2.]." << std::endl;} 
			parameters_good = false;
		}

		if(simParams.dx <= 0.f) {std::cerr << "dx <= 0." << std::endl; return false;}
		if(simParams.dy <= 0.f) {std::cerr << "dy <= 0." << std::endl; return false;}
		if(simParams.dz <= 0.f) {std::cerr << "dz <= 0." << std::endl; return false;}
		

		if(!simParams.residual_reduction) {}

		if(simParams.nx < 1) 
		{
			std::cerr << "nx < 1. There should be at least one row."    << std::endl; 
			parameters_good = false;
		}
		if(simParams.ny < 1) 
		{
			std::cerr << "ny < 1. There should be at least one column." << std::endl; 
			parameters_good = false;
		}
		if(simParams.nz < 3) 
		{
			std::cerr << "nz < 3. There should be at least two slices." << std::endl; 
			parameters_good = false;
		}

		//Limit the domain size
		struct cudaDeviceProp d_info;
		cudaGetDeviceProperties(&d_info, 0); //Current Device should be 0.
		int mx_grid_x = d_info.maxGridSize[0];
		
		// Limited by which kernel? abs_diff.
		int min_thrd_cnt = 128;
		int max_elements = mx_grid_x*min_thrd_cnt;

		// Make sure the size is doable...
		if(domain_size >= max_elements) 
		{
			std::cerr << "The domain size of " << domain_size << " is too big." << std::endl;
			std::cerr << "The limit is " << max_elements << std::endl;
			          
			parameters_good = false;
		}
		
		if(!parameters_good) 
		{
			std::cerr << "Bad parameters found!!" << std::endl;
		}
		
		return parameters_good;
	}

  bool urbModule::validBoundaryMaskQ() const
  {
    bool boundaries_good = true;
    // Check to be sure that urbSetup has been run...
    
    
    // Check against the domain size.
    if (h_bndrs.dim.x*h_bndrs.dim.y*h_bndrs.dim.z != domain_size)
    {
      std::cerr << "Error in " << __FILE__ << ":" << __func__ << std::endl;
      std::cerr << "Host boundary mask matrix dimensions don't match project dims." << std::endl;
      boundaries_good = false;
    }
    
    // Basic values.
    
    // Check the boundary mask entries for valid values.
    for (int i = 0; i < domain_size; i++)
    {
      if (h_bndrs.cmprssd[i] < 0 || 8192 < h_bndrs.cmprssd[i])
      {
        std::cerr << "Error in " << __FILE__ << ":" << __func__ << std::endl;
        std::cerr << "Host boundary condition mask at linear index, " << i;
        std::cerr << ", is outside the acceptable range [1,8192]." << std::endl;
        std::cerr << "Value found: " << h_bndrs.cmprssd[i] << std::endl;
        boundaries_good = false;
        break;
      }
    }
    
    // Check the boiundary mask entries for correct values.
    // TODO 
    
    return boundaries_good;
  }
  
  bool urbModule::validDenominatorsOnDeviceQ() const
  {
    // Run the denoms kernel.
    cudaCompDenoms(d_bndrs, omegarelax, A, B);
    
    // Check the denoms for valid values.
    // Denoms are done on the device...
		cudaMemcpy(h_bndrs.denoms, d_bndrs.denoms, domain_size*sizeof(float), cudaMemcpyDeviceToHost);
		
		int OK_DENOM_COUNT = 8;
		float okDenoms[OK_DENOM_COUNT];
		okDenoms[0] = omegarelax / (2.f*(.5f + A*.5f + B*.5f));
    okDenoms[1] = omegarelax / (2.f*(1.f + A*.5f + B*.5f));
    okDenoms[2] = omegarelax / (2.f*(.5f + A*1.f + B*.5f));
    okDenoms[3] = omegarelax / (2.f*(1.f + A*1.f + B*.5f));
    okDenoms[4] = omegarelax / (2.f*(.5f + A*.5f + B*1.f));
    okDenoms[5] = omegarelax / (2.f*(1.f + A*.5f + B*1.f));
    okDenoms[6] = omegarelax / (2.f*(.5f + A*1.f + B*1.f));
    okDenoms[7] = omegarelax / (2.f*(1.f + A*1.f + B*1.f));
		
    for (int i = 0; i < domain_size; i++)
    {
		  // Only 8 possible values. 2 options for o, 2 options for p, 2 options for q.
		  // o, p and q must each be .5f or 1.f
      // denoms[i] = omegarelax / (2.f*(o[i] + A*p[i] + B*q[i]));
      //h_bndr.denoms[i]
      float denom = h_bndrs.denoms[i];
      bool isOkay = false;
      for (int j = 0; j < OK_DENOM_COUNT; j++)
      {
        // TODO is this error tolerance okay?
        isOkay |= (okDenoms[j] - 1e-6 < denom && denom < okDenoms[j] + 1e-6);
      }
      
      if(!isOkay) 
      {
        std::cerr << "Error in " << __FILE__ << ":" << __func__ << std::endl;
        std::cerr << "Calculated denominator at linear index, " << i;
        std::cerr << ", is not 1 of possible 8 outcomes." << std::endl;
        std::cerr << "Value found: " << denom << std::endl;
        std::cerr << "Possible values: ";
        for (int j = 0; j < OK_DENOM_COUNT; j++)
        {
          std::cerr << okDenoms[j] << "  "; 
        }
        std::cerr << std::endl;
        return false;
      }
    }
    
    return true;
  }

	bool urbModule::validPointersQ
	(
		boundaryMatrices const& bndrs, 
		celltypes const& typs,
		velocities const& vels,
		std::string const& loc,
		bool err
	)
	{
		bool good = true;
		
		if(loc == "host")
		{
			// Boundary Matrices
			if(badPointerQ(bndrs.cmprssd, "cmprssd", loc, err)) {good = false;}
		}
		
		if(loc == "device")
		{
			if(badPointerQ(bndrs.cmprssd, "cmprssd", loc, err)) {good = false;}
			if(badPointerQ(bndrs.denoms,  "denoms",  loc, err)) {good = false;}
		}
		
		// Celltype Matrix
		if(badPointerQ(typs.c, "celltypes", loc, err)) {good = false;}
		
		// Velocity Matrices
		if(badPointerQ(vels.u, "u", loc, err)) {good = false;}
		if(badPointerQ(vels.v, "v", loc, err)) {good = false;}
		if(badPointerQ(vels.w, "w", loc, err)) {good = false;}

		return good;
	}

	float urbModule::enoughDeviceMemoryQ() const
	{

		// Guess this is rough since everything is treated as a float.
		
		//		e,f,g,h,m,n,o,p,q      = 0 			u, v, w    = 3
		//		r, cmprssd, denoms     = 3			visc	     = 1
		//		p1, p2_err, celltypes  = 3	
		//	+	__________________________		+	______________
		
		int domains	=	                 6;			int grids =  4;
		
		// 128 for padding for iter kernel to free x-dim.
		int needed_cell_bytes = (domain_size + 127)*domains*sizeof(float);
		int needed_grid_bytes = grid_size*grids*sizeof(float);
								
		int needed_bytes = needed_cell_bytes + needed_grid_bytes;
		float        needed_MB    = needed_bytes / 1024. / 1024.;
		
		// Originally e, f, g, h, m, n, o, p, q and denoms on device.
		// Now denoms and cmprssd.
		//unsigned int saved_bytes = domain_size*(10*sizeof(float) - 1*sizeof(int) - 1*sizeof(float));
		//float        saved_Mb    = saved_bytes / 1024. / 1024.;

		struct cudaDeviceProp d_info;
		cudaGetDeviceProperties(&d_info, 0); //Current Device should be 0.
		
		int avail_bytes = d_info.totalGlobalMem;
		float        avail_MB    = avail_bytes / 1024. / 1024.;

		float lefto_MB = avail_MB - needed_MB;

		return (lefto_MB < 0) ? lefto_MB : needed_MB ;
	}

	void urbModule::allocateDeviceMemory() 
	{
		// Prevent leaks.
		if(this->validDevicePointersQ(false)) {this->deallocateDeviceMemory();}
	
		float enough_device_mem = this->enoughDeviceMemoryQ();
		std::ostringstream oss;
		std::fixed(oss);
		oss.precision(2);
		oss << "Allocating device memory of " << enough_device_mem << " Mb...";
		qwrite(oss.str());

			stpwtchs->malloc->start();
	
		if(enough_device_mem <= 0) 
		{
		  qwrite("done\n");
			std::cerr << "Not enough memory on device." << std::endl;
			std::cerr << "Device memory left: " << enough_device_mem << "." << std::endl;
			
			exit(EXIT_FAILURE);
			// Set don't do anything flag...
			
			return;
		} 


		// Padding added to free x-dim in the iteration kernel.
		int padded_size = (domain_size + 127);
		int cll_data_size = padded_size*sizeof(int);
		int dmn_data_size = padded_size*sizeof(float);
		int typ_data_size = padded_size*sizeof(CellType);

		// Matrices with cell centered values
		cudaMalloc((void**) &d_bndrs.cmprssd, cll_data_size);	
			showError("urbModule::allocateDeviceMemory() - cmprssd");
    cudaMalloc((void**) &d_bndrs.denoms, dmn_data_size); 
      showError("urbModule::allocateDeviceMemory() - denoms");

		cudaMalloc((void**) &d_typs.c, typ_data_size); 	
			showError("urbModule::allocateDeviceMemory() - d_typs");

		cudaMalloc((void**) &d_r, dmn_data_size);		
			showError("urbModule::allocateDeviceMemory() - d_r");

		cudaMalloc((void**) &d_p1, dmn_data_size);		
			showError("urbModule::allocateDeviceMemory() - d_p1");//new iter
		cudaMalloc((void**) &d_p2_err, dmn_data_size); 	
			showError("urbModule::allocateDeviceMemory() - d_p2_err");//old iter

		cudaMalloc((void**) &d_abse, sizeof(float));	
			showError("urbModule::allocateDeviceMemory() - d_abse");
		
		// Matrices with grid point values 
		// u,v,w and uo,vo,wo are one larger than domain. 
		// Values at edges rather than centers.
		    padded_size   = (grid_size + 127);
		int grd_data_size = padded_size * sizeof(float);

		cudaMalloc((void**) &d_vels.u, grd_data_size); 		
			showError("urbModule::allocateDeviceMemory() - d_u");
		cudaMalloc((void**) &d_vels.v, grd_data_size); 		
			showError("urbModule::allocateDeviceMemory() - d_v");
		cudaMalloc((void**) &d_vels.w, grd_data_size); 		
			showError("urbModule::allocateDeviceMemory() - d_w");

		cudaMalloc((void**) &d_visc, grd_data_size); 	
			showError("urbModule::allocateDeviceMemory() - d_visc");

		cudaThreadSynchronize();
		
		this->validDevicePointersQ();
		
			stpwtchs->malloc->stop();
			
		qwrite("done.\n");
	}

	void urbModule::deallocateDeviceMemory()
	{
		qwrite("urbModule deallocating device memory...");
		
		cudaFree(d_bndrs.cmprssd); d_bndrs.cmprssd = 0;
		cudaFree(d_bndrs.denoms);  d_bndrs.denoms  = 0;

		cudaFree(d_typs.c); d_typs.c = 0;

		cudaFree(d_r); d_r = 0;

		cudaFree(d_vels.u);  d_vels.u = 0;
		cudaFree(d_vels.v);  d_vels.v = 0;
		cudaFree(d_vels.w);  d_vels.w = 0;

		cudaFree(d_visc); d_visc = 0;

		cudaFree(d_p1);	    d_p1     = 0;
		cudaFree(d_p2_err);	d_p2_err = 0;
		cudaFree(d_abse);   d_abse   = 0;
		
		qwrite("done.\n");
	}
	
	void urbModule::allocateHostMemory()
	{
		// Prevent leaks.
		if(this->validHostPointersQ(false)) {this->deallocateHostMemory();}
		
		qwrite("Allocating host memory...");
	
	    stpwtchs->malloc->start();
	
	
		// Get memory available
		// Is it possible?
		
		// Calculate needed memory
		// 0 matrices of floats at domain_size : e, f, g, h, m, n, o, p, q
		// 3 matrices of floats at grid_size   : uo, vo, wo
		// 2 matrix of integers at domain_size : celltypes, cmprssd
		
		int domain_size_bytes = domain_size*(0*sizeof(float) + 2*sizeof(int));
		int grid_size_bytes   = grid_size*3*sizeof(float);
		
		int needed_bytes = domain_size_bytes + grid_size_bytes;
		float        needed_MB    = needed_bytes / 1024. / 1024.;
			
		// Report the needed memory
		std::ostringstream oss;
		std::fixed(oss);
		oss.precision(2); 
		oss << "needed host memory: " << needed_MB << " Mb...";
		qwrite(oss.str());
		
		h_bndrs.cmprssd = new      int[domain_size];
		h_typs.c        = new CellType[domain_size];

		h_ntls.u  = new float[grid_size];
		h_ntls.v  = new float[grid_size];
		h_ntls.w  = new float[grid_size];

	/*
		size_t grid_bytes = grid_size*sizeof(float);

		cudaHostAlloc((void**) &h_ntls.u, grid_bytes, cudaHostAllocMapped);
			showError("urbModule::allocateHostMemory() - h_u");
		cudaHostAlloc((void**) &h_ntls.v, grid_bytes, cudaHostAllocMapped);
			showError("urbModule::allocateHostMemory() - h_v");
		cudaHostAlloc((void**) &h_ntls.w, grid_bytes, cudaHostAllocMapped);
			showError("urbModule::allocateHostMemory() - h_w");
		
		this->validHostPointersQ();
		
		cudaHostGetDevicePointer((void**) &d_ntls.u, h_ntls.u, 0);
		cudaHostGetDevicePointer((void**) &d_ntls.v, h_ntls.v, 0);
		cudaHostGetDevicePointer((void**) &d_ntls.w, h_ntls.w, 0);
	*/
		
			stpwtchs->malloc->stop();
			
		qwrite("done.\n");
	}
	
	void urbModule::deallocateHostMemory()
	{
		qwrite("urbModule deallocating host memory...");

		delete [] h_bndrs.cmprssd;
		delete [] h_typs.c;
		
		delete [] h_ntls.u;
		delete [] h_ntls.v;
		delete [] h_ntls.w;
		
		qwrite("done.\n");
	}

  void urbModule::allocateHostRunnableMemory()
  {
    qwrite("Allocating HostRunnableMemory...");
  
    if(!host_runnable) {return;}
    
	  // Prevent leaks.
	  //this->deallocateHostRunnableMemory();
		
		qwrite("Allocating memory to run SOR on host...");
	
		  stpwtchs->malloc->start();
		
		// Calculate needed memory
		// 3 matrices of floats at domain_size : e, f, g, h, m, n, o, p, q
		// 3 matrices of floats at grid_size : u, v, w
		int domain_size_bytes = domain_size*3*sizeof(float);
		int grid_size_bytes   = grid_size*3*sizeof(float);
		
		int needed_bytes = domain_size_bytes + grid_size_bytes;
		float        needed_MB    = needed_bytes / 1024. / 1024.;

		// Report the needed memory
		std::ostringstream oss; 
		std::fixed(oss);
		oss.precision(2);
		oss << "addition host memory needed: " << needed_MB << "Mb...";
		qwrite(oss.str());
		
		h_bndrs.denoms = new float[domain_size];
		
		h_vels.u = new float[grid_size];
		h_vels.v = new float[grid_size];
		h_vels.w = new float[grid_size];
		
		h_r      = new float[domain_size];
		h_p1     = new float[domain_size];
		h_p2_err = new float[domain_size];
		
		  stpwtchs->malloc->stop();
		
		qwrite("done.\n");
  }
  
	void urbModule::deallocateHostRunnableMemory()
	{
	  qwrite("Deallocating host-runnable memory...");
	
	  delete [] h_bndrs.denoms;
	
	  delete [] h_vels.u;
	  delete [] h_vels.v;
	  delete [] h_vels.w;
	  
	  delete [] h_r;
	  delete [] h_p1;
	  delete [] h_p2_err;
	  
	  qwrite("done.\n");
	}

	void urbModule::transferDataToDevice() 
	{
		qwrite("Transferring data to device...");
	
			stpwtchs->trnsfr->start();
	
		if(!this->validDevicePointersQ())
		{
			std::cerr 
			<< 
				"urbModule::Bad device pointers found while trying to transfer." 
			<< 
			std::endl;
		}
		if(!this->validHostPointersQ())
		{
			std::cerr
			<<
				"urbModule::Bad host pointers found while trying to transfer."
			<<
			std::endl;
		}

		// Matrices with cell centered values
		int int_data_size = domain_size * sizeof(int);
		int typ_data_size = domain_size * sizeof(CellType); 
		
		std::string prefix = "urbModule::transferDataToDevice() - ";

    // Transfering boundaries to device is needed with init by fortran, otherwise not.		
		cudaMemcpy(d_bndrs.cmprssd, h_bndrs.cmprssd, int_data_size, cudaMemcpyHostToDevice);
			showError((prefix + "d_cmprssd").c_str());
		cudaMemcpy(d_typs.c, h_typs.c, typ_data_size, cudaMemcpyHostToDevice);		
			showError((prefix + "d_typs").c_str());
		// End cell centered matrices

		// Matrices with grid point values
		int grid_data_size = grid_size * sizeof(float);

		cudaMemcpy(d_vels.u, h_ntls.u, grid_data_size, cudaMemcpyHostToDevice);	
			showError((prefix + "d_u").c_str());
		cudaMemcpy(d_vels.v, h_ntls.v, grid_data_size, cudaMemcpyHostToDevice);	
			showError((prefix + "d_v").c_str());
		cudaMemcpy(d_vels.w, h_ntls.w, grid_data_size, cudaMemcpyHostToDevice);	
			showError((prefix + "d_w").c_str());
		// End grid point matrices

		cudaThreadSynchronize();

			stpwtchs->trnsfr->stop();

		qwrite("done.\n");
	}

	bool urbModule::validDevicePointersQ(bool err) const
	{
		bool good = urbModule::validPointersQ
												(
													d_bndrs, 
													d_typs, 
													d_vels, 
													"device",
													err
												);
		
		if(badPointerQ(d_r,      "r",      "device", err)) {good = false;}
		if(badPointerQ(d_p1,     "p1",     "device", err)) {good = false;}
		if(badPointerQ(d_p2_err, "p2_err", "device", err)) {good = false;}
		if(badPointerQ(d_abse,   "abse",   "device", err)) {good = false;}
		if(badPointerQ(d_visc,   "visc",   "device", err)) {good = false;}
				
		return good;
	}

	bool urbModule::validHostPointersQ(bool err) const
	{
		return urbModule::validPointersQ(h_bndrs, h_typs, h_ntls, "host", err);
	}

	bool urbModule::anyOutputQ() const
	{
		return 
			output_velocities  || 
			output_boundaries  || 
			output_divergence  || 
			output_lagrangians ||
			output_celltypes
		;
	}

	bool urbModule::badPointerQ
	(
		const void* ptr, 
		std::string const& name, 
		std::string const& loc,
		bool err
	)
	{
		if(ptr == NULL)
		{
			if(err)
			{
				std::cerr 
				<< 
					"Pointer to " << name << " on " << loc << " is null." 
				<< 
				std::endl;
			}
			return true;
		}
		else
		{
			return false;
		}
	}
}

