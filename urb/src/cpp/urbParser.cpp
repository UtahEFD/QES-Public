#include "urbParser.h"

#include <iostream>
#include <iomanip>

#include "quicloader/legacyFileParser.h"
#include "quicloader/QUBuildings.cpp"
#include "quicloader/QUFileOptions.cpp"
#include "quicloader/QUICProject.cpp"
#include "quicloader/QUMetParams.cpp"
#include "quicloader/QUSimparams.cpp"
#include "quicloader/QUSensor.cpp"
#include "quicloader/QUVelocities.cpp"
#include "quicloader/standardFileParser.h"

#include "building.h"
#include "../util/matrixIO.h"

namespace QUIC
{
	void urbParser::parse(urbModule* um, std::string filepath)
 	{
            um->stpwtchs->parse->start();

            // This loads in the entire QUIC project (or should)
            QUICProject quicProject = QUICProject(filepath, !um->isQuietQ(), true);
            um->input_directory = quicProject.m_quicProjectPath;
    
            // Get the basic simulation information.
            um->simParams = quicProject.quSimParamData;
            um->setDimensions(um->simParams.nx, um->simParams.ny, um->simParams.nz);
            um->sim_params_parsed = true;

            // Get the sensor information.
            um->metParams = quicProject.quMetParamData;

            // Metparams contains the sensors...
            // currently, QUIC loader only handles a single sensor for
            // the MetParams....
            // quSensorParams* tmp = new quSensorParams();
            // tmp->siteName  = um->metParams.siteName;
            // tmp->fileName  = um->metParams.sensorFileName;
            um->sensors.push_back( um->metParams.quSensorData );
            // for(unsigned int i = 0; i < um->sensors.size(); i++)
            // {
            // um->sensors[0]->beVerbose = !(um->isQuietQ());
                // um->sensors[0]->readQUICFile(um->input_directory + um->sensors[i]->fileName); 
                //um->sensors[i]->print();    

            // Only handle 1 sensor now...
            // Some sensor setup. Probably shouldn't stay here...
            um->sensors[0].prfl_lgth = um->gz;
            um->sensors[0].u_prof = new float[um->gz];
            um->sensors[0].v_prof = new float[um->gz];
            // }

            // Getting the buildings information
            um->fileBuildings = quicProject.quBuildingData;
            // Take the building file information and create QUIC Urb usable building objects.
            for (unsigned i = 0; i < um->fileBuildings.buildings.size(); i++)
            {
                um->buildings.push_back(urbBuilding(um->fileBuildings.buildings[i]));
            }
		
            um->initialize(); // Needed for parse sensors, which requires gz.

            um->stpwtchs->parse->stop();
	}

	void urbParser::dump
	(
		urbModule* um,
		std::string drctry, 
		std::string prefix //= "um_"
	)
	{
		if(!um->anyOutputQ()) {return;}
	
		if(drctry.at(drctry.length() - 1) != '/') {drctry = drctry.append("/");}
		
		um->qwrite("Dumping " + um->name + " from " + um->input_directory + " to " + drctry + "\n"); 
		
		if(um->output_celltypes)    {QUIC::urbParser::outputCellTypes   (um, drctry, prefix);}
		if(um->output_boundaries)   {QUIC::urbParser::outputBoundaries  (um, drctry, prefix);}
		if(um->output_divergence)   {QUIC::urbParser::outputDivergence  (um, drctry, prefix);}
    if(um->output_denominators) {QUIC::urbParser::outputDenominators(um, drctry, prefix);}
    if(um->output_lagrangians)  {QUIC::urbParser::outputLagrangians (um, drctry, prefix);}
		if(um->output_velocities)   {QUIC::urbParser::outputVelocities  (um, drctry, prefix);}
		if(um->output_viscocity)    {QUIC::urbParser::outputViscocity   (um, drctry, prefix);}
	}

  void urbParser::generatePlumeInput(QUIC::urbModule* um, std::string drctry)
  {
    if(drctry.at(drctry.length() - 1) != '/') {drctry = drctry.append("/");}
    
    bool generated = false;
    
    generated  = QUIC::urbParser::output_QU_velocities (um, drctry);
		generated ^= QUIC::urbParser::output_QU_celltypes  (um, drctry);
		generated ^= QUIC::urbParser::output_PlumeInputFile(um, drctry);
		
		if(!generated)
		{
      std::cerr << "Error::Unable to generate Plume Input files." << std::endl;
    }
  }
		
	void urbParser::outputInitialVelocityData
	(
		urbModule* um,
		std::string drctry,
		std::string const& prefix //= "um_"
	)
	{
		if(drctry.at(drctry.length() - 1) != '/') {drctry = drctry.append("/");}
		
		std::string path = drctry + prefix + "uofield.dat";
		std::fstream vdata(path.c_str(), std::fstream::out);
		
		if(!vdata.is_open()) 
		{
			std::cerr << "Unable to open " << path << ", nothing output." << std::endl; 
			return;
		}

		vdata.precision(5);
		vdata << std::fixed;

		//Print the header information similiar to outfile.f90.
		vdata << " % Inital velocity field i,j,k,uo,vo,wo" << std::endl;
		vdata << std::setw(9) << "1"   << std::setw(7) << " " << "!Number of time steps" << std::endl; // Limited to one time step.
		vdata << " % Begin Output for new time step" << std::endl;
		vdata << std::setw(9) << "1"   << std::setw(7) << " " << "!Time Increment" << std::endl;  // Time step
		vdata << std::setw(9) << 0.0   << std::setw(7) << " " << "!Time" << std::endl; // Time

		int gx = um->h_ntls.dim.x;
		int gy = um->h_ntls.dim.y;
		int gz = um->h_ntls.dim.z;
		
		// Allocate memory on the host
		float* h_u = new float[um->grid_size];
		float* h_v = new float[um->grid_size];
		float* h_w = new float[um->grid_size];

		// copy initial velocity data to h_u, h_v and h_w.
		for(int c = 0; c < gx*gy*gz; c++)
		{
			h_u[c] = um->h_ntls.u[c];
			h_v[c] = um->h_ntls.v[c];
			h_w[c] = um->h_ntls.w[c];
		}

		// Print the velocity information. (Currently only one time step...)
		// Not staggered data output
		int cI;

		for(int k = 0; k < gz; k++) 
		for(int j = 0; j < gy; j++) 
		for(int i = 0; i < gx; i++) 
		{
			cI = k*gy*gx + j*gx + i;

			vdata << std::setw(6)  << (i + 1);
			vdata << std::setw(6)  << (j + 1);
			vdata << std::setw(6)  << (k + 1);
			
			vdata << std::setw(18) << h_u[cI];
			vdata << std::setw(18) << h_v[cI];
			vdata << std::setw(18) << h_w[cI];
			
			vdata << std::endl;
		}
		
		vdata << std::endl;
		vdata.close();

		delete [] h_u; delete [] h_v; delete [] h_w;
	}

	void urbParser::printValues(QUIC::urbModule* um, std::ostream& out)
	{
		using namespace std;
		
		out << endl << endl;
		out << " --- Sim Parameters --- " << endl;
		out << "\tstart_time = " << um->simParams.start_time << endl;
		out << "\ttime_incr  = " << um->simParams.time_incr  << endl;
		out << "\tnum_time_steps = " << um->simParams.num_time_steps << endl;
		out << "\troof_type   = " << um->simParams.roof_type   << endl;
		out << "\tupwind_type = " << um->simParams.upwind_type << endl;
		out << "\tcanyon_type = " << um->simParams.canyon_type << endl;
		out << "\tintersection_flag = " << um->simParams.intersection_flag << endl;
		out << "\tdomain_rotation = " << um->simParams.domain_rotation << endl;
		out << "\tutmx = " << um->simParams.utmx << endl;
		out << "\tutmy = " << um->simParams.utmy << endl;
		out << "\tutm_zone = " << um->simParams.utm_zone << endl;
		out << "\tquic_cfd_type = " << um->simParams.quic_cfd_type << endl;
		out << "\twake_type     = " << um->simParams.wake_type     << endl;
		out << endl << endl;
			
		out << " --- Met Parameters --- " << endl;
		out << "\tmet_input_type  = " << um->metParams.metInputFlag  << endl;
		out << "\tnum_sites       = " << um->metParams.numMeasuringSites << endl;
		out << "\tnum_vert_points = " << um->metParams.maxSizeProfiles << endl;
		for(unsigned int i = 0; i < um->sensors.size(); i++)
		{
			out << "\t\t" << flush; um->sensors[i].print();
		}
		out << endl << endl;
			
		out << " --- Buildings --- " << endl;
		out << "\tx_subdomain_start = " << um->fileBuildings.x_subdomain_sw << endl;
		out << "\ty_subdomain_start = " << um->fileBuildings.y_subdomain_sw << endl;
		out << "\tx_subdomain_end = " << um->fileBuildings.x_subdomain_ne << endl;
		out << "\ty_subdomain_end = " << um->fileBuildings.y_subdomain_ne << endl;
		out << "\tzo = " << um->buildings[0].zo << endl;
		out << "\tnumbuilds = " << um->buildings.size() << endl;
		for(unsigned int i = 0; i < um->buildings.size(); i++)
		{
			out << "\t\t" << flush; um->buildings[i].print();
		}
		out << endl << endl;
			
		out << " --- File Options --- " << endl;
		out << "\toutput_format_type    = " << um->fileOptions.format_type    << endl;
    out << "\toutput_uofield_flag   = " << um->fileOptions.uofield_flag   << endl;
    out << "\toutput_uosensor_flag  = " << um->fileOptions.uosensor_flag  << endl;
		out << "\toutput_staggered_flag = " << um->fileOptions.staggered_flag << endl;
		out << endl << endl;
		
		out << " --- urbModule variables --- " << endl;
		out << "\tnx = " << um->simParams.nx << endl;
		out << "\tny = " << um->simParams.ny << endl;
		out << "\tnz = " << um->simParams.nz << endl;

		out << "\tdx = " << um->simParams.dx << endl;
		out << "\tdy = " << um->simParams.dy << endl;
		out << "\tdz = " << um->simParams.dz << endl;

		out << "\tmax_iterations     = " << um->simParams.max_iterations     << endl;
		out << "\tresidual_reduction = " << um->simParams.residual_reduction << endl;

		out << "\tdiffusion_flag = " << um->simParams.diffusion_flag << endl;
		out << "\tdiffusion_step = " << um->simParams.diffusion_step << endl;
		out << endl << endl;
	}

	void urbParser::printSetupTimes(QUIC::urbModule* um, std::ostream& out)
	{
		using namespace std;
		
		out << fixed;
		out.precision(6);
	
		out << endl;
		out << "<< Setup Timings >> " << endl;
		
		out << *um->stpwtchs->parse     << endl;
		out << *um->stpwtchs->init      << endl;
		out << *um->stpwtchs->sort      << endl;
		out << *um->stpwtchs->sensor    << endl;
		out << *um->stpwtchs->bldngprm  << endl;
		out << *um->stpwtchs->intersect << endl;
		out << *um->stpwtchs->bndrymat  << endl;
	}

	void urbParser::printIterTimes(QUIC::urbModule* um, std::ostream& out)
	{
		using namespace std;
	
		out << endl;
		out << "Dimensions: " << um->simParams.nx << "x" << um->simParams.ny << "x" << um->simParams.nz << endl;
		out << "Iterations: " << um->iteration << " (" << um->simParams.max_iterations << ")" << endl;
		out << "Omegarelax: " << um->omegarelax << endl;
		
		out << fixed;
		out.precision(6);
		
		if(um->runto_eps > 0.) 
		{
			out << "Error (runto_eps) : " << um->runto_eps << endl;
		}
		else
		{
			out << "Error (eps - residual) : " << um->eps << endl;
		}
	  
	  double comput_time = um->stpwtchs->comput->getElapsedTime();
	  double denoms_time = um->stpwtchs->denoms->getElapsedTime();
	  double total_time  = um->stpwtchs->getIterationTime();
	  
		float avg_iter = (comput_time + denoms_time) / um->getIteration();
		
		out << endl;
		out << "<< Iteration Timings >> " << endl;
		
		out << *um->stpwtchs->diverg << endl;
		out << *um->stpwtchs->denoms << endl;
		out << *um->stpwtchs->comput << endl;
		out << *um->stpwtchs->euler  << endl;
		out << *um->stpwtchs->diffus << endl;
		
		out << "Iter Total  : " << total_time << " secs." << endl;
		out << "Iter Avg    : " << avg_iter << " secs." << endl;
	}

	void urbParser::printInfo(QUIC::urbModule* um, std::ostream& out)
	{
		QUIC::urbParser::printSetupTimes(um, out);
		QUIC::urbParser::printIterTimes(um, out);		
		
		out << std::endl;
		out << "Total time : " << um->stpwtchs->getTotalElapsedTime() << " secs" << std::endl;
	}

  void urbParser::outputCellTypes
	(
		const urbModule* um,
		std::string const& drctry,
		std::string const& prefix /*= "um_"*/
	)
	{
		um->qwrite("  Outputting celltype data...");
	
	  // if(drctry.at(drctry.length() - 1) != '/') {drctry = drctry.append("/");}
		int nx = um->d_typs.dim.x;
		int ny = um->d_typs.dim.y;
		int nz = um->d_typs.dim.z;
	
	  CellType* types = QUIC::urbModule::getCUDAdata<CellType>(um->d_typs.c, nx, ny, nz);
	  int* data = new int[um->domain_size];
	  for(int i = 0; i < um->domain_size; i++) data[i] = (int) types[i];
	  
	  if(data)
		{
		  std::string file = drctry + prefix + "celltype.dat";
		  outputMatrix(file.c_str(), data, nx, ny, nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output celltype data, no device pointer." << std::flush;
		}
	
		um->qwrite("done.\n");
	}
	
	void urbParser::outputBoundaries
	(
		const urbModule* um,
		std::string const& drctry,
		std::string const& prefix /*= "um_"*/
	)
	{
		um->qwrite("  Outputting boundary matrix data...");
	
		int nx = um->h_bndrs.dim.x;
		int ny = um->h_bndrs.dim.y;
		int nz = um->h_bndrs.dim.z;
	
		std::string pre = drctry + prefix;
	
	/*
		std::string files[] = 
		{
			pre + "e.dat", pre + "f.dat",
			pre + "g.dat", pre + "h.dat",
			pre + "m.dat", pre + "n.dat",
			pre + "o.dat", pre + "p.dat",
			pre + "q.dat"
		};
	
		float* b_mats[] = 
		{
			um->h_bndrs.e, um->h_bndrs.f,
			um->h_bndrs.g, um->h_bndrs.h,
			um->h_bndrs.m, um->h_bndrs.n,
			um->h_bndrs.o, um->h_bndrs.p,
			um->h_bndrs.q
		};
				
		for(int i = 0; i < 9; i++)
		{
			outputMatrix(files[i].c_str(), b_mats[i], nx, ny, nz);
		}
	*/
	
	  int* data = QUIC::urbModule::getCUDAdata<int>(um->d_bndrs.cmprssd, nx, ny, nz);
	
	  if(data)
		{
		  std::string file = drctry + prefix + "d_bndr.dat";
		  outputMatrix(file.c_str(), data, nx, ny, nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output compressed boundary data, no device pointer." << std::flush;
		}
	
		um->qwrite("done.\n");
	}

  void urbParser::outputDivergence
  (
    const urbModule* um,
    std::string const& drctry,
    std::string const& prefix
  )
  {
		um->qwrite("  Outputting divergence data...");

		float* data = QUIC::urbModule::getCUDAdata(um->d_r, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		
		if(data)
		{
		  std::string file = drctry + prefix + "r.dat";
		  outputMatrix(file.c_str(), data, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output divergence data, no device pointer." << std::flush;
		}
		
		um->qwrite("done.\n");
  }
  
  void urbParser::outputDenominators
  (
    const urbModule* um,
    std::string const& drctry,
    std::string const& prefix
  )
  { 
		um->qwrite("  Outputting denominators data...");
		
		float* data = QUIC::urbModule::getCUDAdata<float>(um->d_bndrs.denoms, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		
		if(data)
		{
		  std::string file = drctry + prefix + "denoms.dat";
		  outputMatrix(file.c_str(), data, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output denominators data, no device pointer." << std::flush;
		}
		
		um->qwrite("done.\n");
  }
  
  void urbParser::outputLagrangians
  (
    const urbModule* um,
    std::string const& drctry,
    std::string const& prefix
  )
	{
		um->qwrite("  Outputting lagrangian data...");
		
		float* data = QUIC::urbModule::getCUDAdata<float>(um->d_p1, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		
		if(data)
		{
		  std::string file = drctry + prefix + "p1.dat";
		  outputMatrix(file.c_str(), data, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output p1 - Lagrangians, no device pointer." << std::flush;
		}
		
		um->qwrite("done.\n");
	}
  
  void urbParser::outputErrors
	(
	  const urbModule* um, std::string const& drctry,
	  std::string const& prefix
	)
	{
	  um->qwrite("  Outputting error data...");
		
		float* data = QUIC::urbModule::getCUDAdata<float>(um->d_p2_err, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		
		if(data)
		{
		  std::string file = drctry + prefix + "p2_err.dat";
		  outputMatrix(file.c_str(), data, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output errors, no device pointer." << std::flush;
		}
		
		um->qwrite("done.\n");
	}
  	
	void urbParser::outputVelocities
	(
		const urbModule* um,
		std::string const& drctry,
		std::string const& prefix /*= "um_"*/
	)
	{
		um->qwrite("  Outputting velocity data...");
	
		int gx = um->h_ntls.dim.x;
		int gy = um->h_ntls.dim.y;
		int gz = um->h_ntls.dim.z;
		
		std::string pre = drctry + prefix;
		
		outputMatrix((pre + "uo.dat").c_str(), um->h_ntls.u, gx, gy, gz);
		outputMatrix((pre + "vo.dat").c_str(), um->h_ntls.v, gx, gy, gz);
		outputMatrix((pre + "wo.dat").c_str(), um->h_ntls.w, gx, gy, gz);

    float* u = QUIC::urbModule::getCUDAdata<float>(um->d_vels.u, gx, gy, gz);
    float* v = QUIC::urbModule::getCUDAdata<float>(um->d_vels.v, gx, gy, gz);
		float* w = QUIC::urbModule::getCUDAdata<float>(um->d_vels.w, gx, gy, gz);
		
		outputMatrix((pre + "u.dat").c_str(), u, gx, gy, gz);
		outputMatrix((pre + "v.dat").c_str(), v, gx, gy, gz);
		outputMatrix((pre + "w.dat").c_str(), w, gx, gy, gz);

    delete [] u; delete [] v; delete [] w;		
		
		um->qwrite("done.\n");
	}
	
  void urbParser::outputViscocity
  (
    const urbModule* um,
    std::string const& drctry,
    std::string const& prefix
  )
	{
		um->qwrite("  Outputting viscocity data...");
		
		float* data = QUIC::urbModule::getCUDAdata<float>(um->d_visc, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		
		if(data)
		{
		  std::string file = drctry + prefix + "visc.dat";
		  outputMatrix(file.c_str(), data, um->simParams.nx, um->simParams.ny, um->simParams.nz);
		  delete [] data;
		}
		else
		{
		  std::cerr << "Unable to output viscocity data, no device pointer." << std::flush;
		}
		
		um->qwrite("done.\n");
	}

	bool urbParser::parse_input(QUIC::urbModule* um, std::string const& drctry)
	{
		um->qwrite("Parsing input.txt...");
			
		std::string filepath = drctry + "input.txt";
		std::ifstream file(filepath.c_str(), std::ifstream::in);
		if(!file.is_open())
		{
			um->qwrite("urbParser could not open :: " + filepath + ".\n");
			return false;
		}
		else
		{
			file.close();
		}
	
		standardFileParser* sfp = new standardFileParser(); 
		
		// Describe the elements you're looking for.
		boolElement  be_quiet      = boolElement ("quiet");
		boolElement  be_use_fort   = boolElement ("use_fortran");

		boolElement  be_out_typs   = boolElement ("output_celltypes");
		boolElement  be_out_bndr   = boolElement ("output_boundaries");
		boolElement  be_out_dvrg   = boolElement ("output_divergence");
		boolElement  be_out_dnms   = boolElement ("output_denominators");
		boolElement  be_out_lgng   = boolElement ("output_lagrangians");
		boolElement  be_out_vels   = boolElement ("output_velocities");
		boolElement  be_out_visc   = boolElement ("output_viscocity");
		
		boolElement  be_diff_flag	 = boolElement ("diffusion_flag");
		intElement   ie_iter_step  = intElement  ("iteration_step");
		floatElement fe_runto_eps  = floatElement("runto_epsilon");
		floatElement fe_omegarelax = floatElement("omegarelax");
		
		// Commit them to the parser.
		sfp->commit(be_quiet);
		sfp->commit(be_use_fort);
		
		sfp->commit(be_out_typs);		
		sfp->commit(be_out_bndr);
		sfp->commit(be_out_dvrg);
		sfp->commit(be_out_dnms);
		sfp->commit(be_out_lgng);
		sfp->commit(be_out_vels);
		sfp->commit(be_out_visc);
		
		sfp->commit(be_diff_flag);
		sfp->commit(ie_iter_step);
		sfp->commit(fe_runto_eps);
		sfp->commit(fe_omegarelax);
		
		// Should look for input.txt in the running drctry.
		sfp->study(drctry + "input.txt");

		// See if they were found.
		if(sfp->recall(be_quiet))     {um->quiet           = be_quiet.value;}
		if(sfp->recall(be_use_fort))  {um->use_fortran     = be_use_fort.value;}
		
		if(sfp->recall(be_out_typs))  {um->output_celltypes    = be_out_typs.value;}
		if(sfp->recall(be_out_bndr))  {um->output_boundaries   = be_out_bndr.value;}
		if(sfp->recall(be_out_dvrg))  {um->output_divergence   = be_out_dvrg.value;}
		if(sfp->recall(be_out_dnms))  {um->output_denominators = be_out_dnms.value;}
		if(sfp->recall(be_out_lgng))  {um->output_lagrangians  = be_out_lgng.value;}
		if(sfp->recall(be_out_vels))  {um->output_velocities   = be_out_vels.value;}
		if(sfp->recall(be_out_visc))  {um->output_viscocity    = be_out_visc.value;}

		if(sfp->recall(be_diff_flag)) {um->simParams.diffusion_flag  = be_diff_flag.value;}
		// Call these to ensure proper parameter values.
		if(sfp->recall(ie_iter_step))  {um->setIterationStep(ie_iter_step.value);}
		if(sfp->recall(fe_runto_eps))  {um->setEpsilon      (fe_runto_eps.value);}
		if(sfp->recall(fe_omegarelax)) {um->setOmegaRelax   (fe_omegarelax.value);}

		delete sfp;
		
		um->qwrite("done.\n");
		
		// \\todo more checking for correct parsing.
		return true;
	}
	
	// \\todo get a common parser working on this stuff --> change the formatting
	void urbParser::output_QP_buildout(urbModule* um, std::string const& drctry)
	{
		// ???
	}
	
	bool urbParser::output_QU_velocities (urbModule* um, std::string const& drctry)
	{
	
	  // This allocates memory.
    velocities finalVels = um->getFinalVelocities();
	  quVelocities quVels = quVelocities();

	  std::string filepath = drctry + "QU_velocity.dat";	  
    quVels.writeQUICFile(filepath, finalVels, um->simParams);

    delete [] finalVels.u;
    delete [] finalVels.v;
    delete [] finalVels.w;

	  return true; // Check for file errors...
	}
	
	bool urbParser::output_QU_celltypes(urbModule* um, std::string const& drctry)
	{
	  um->qwrite("Writing QU_celltypes.dat...");
	
	  using namespace std;
	
	  string filepath = drctry + "QU_celltype.dat";
	  ofstream writeCell(filepath.c_str());
	  if(!writeCell.is_open()) 
	  {
		  cerr << "failed to open " + filepath + "." << flush;
		  return false;
	  }

    celltypes clltyps = um->getCellTypes();
    
    int row = clltyps.dim.x;
    int slc = clltyps.dim.x*clltyps.dim.y;

    for(int k = 0; k < clltyps.dim.z; k++)
    for(int j = 0; j < clltyps.dim.y; j++)
    for(int i = 0; i < clltyps.dim.x; i++)
    {
      int ndx = k*slc + j*row + i;
      
      // Write out indices.
      writeCell << setw(12) << (i + .5);
      writeCell << setw(12) << (j + .5);
      writeCell << setw(12) << (k - .5);
       
      // Write out celltype value.
	    writeCell << setw(12) << clltyps.c[ndx];
	    writeCell << endl;
	  }

	  writeCell.close();
	  
	  um->qwrite("done.\n");
	  
	  return true;
	}
	
	bool urbParser::output_PlumeInputFile(urbModule* um, std::string const& drctry)
	{
    um->qwrite("Writing Plume:: input.txt...");

    using namespace std;

	  string filepath = drctry + "input.txt";
	  ofstream writeInput(filepath.c_str());
	  if(!writeInput.is_open()) 
	  {
		  cerr << "Failed to open " + filepath + "." << endl;
		  return false;
	  }

	  for(int i = 0; i < 5; i++) writeInput << "filler" << endl;
	  
	  writeInput << "nx " << um->simParams.nx << endl;
	  writeInput << "ny " << um->simParams.ny << endl;
	  writeInput << "nz " << um->simParams.nz << endl;

	  for(int i = 0; i < 10; i++) writeInput << "filler" << endl;

    writeInput << "output_file " << "data.txt" << endl; // Use something else?

	  for(int i = 0; i < 33; i++) writeInput << "filler" << endl;

	  writeInput << "numBuild " << um->buildings.size() << endl;
	  for(unsigned i = 0; i < um->buildings.size(); i++)
	  {
	    writeInput << "build " << flush;
	    writeInput << um->buildings[i].xfo << " " << flush;
	    writeInput << um->buildings[i].yfo << " " << flush;
	    writeInput << um->buildings[i].zfo << " " << flush;
	    
	    writeInput << um->buildings[i].height << " " << flush;
	    writeInput << um->buildings[i].width << " " << flush;
	    writeInput << um->buildings[i].length << " " << flush;
	  }
	  
	  writeInput.close();
	  
	  um->qwrite("done.\n");
	  
	  return true;
	}
}
