#include "urbTimer.h"

namespace QUIC 
{
	urbTimer::urbTimer() : urbModule::urbModule()
	{		
		start.x = start.y = start.z = 0;
		end.x   = end.y   = end.z   = 0;
		step.x  = step.y  = step.z  = 0;
		tDim.x  = tDim.y  = tDim.z  = 0;
	}

	urbTimer::~urbTimer() {}

	void urbTimer::setRange(int3 const& s, int3 const& e)
	{
		if(s.x >= 64 && s.x % 64 == 0 && s.y >= 1 && s.z >= 3) {start = s;}
		else 
		{
			start.x = simParams.nx;
			start.y = simParams.ny;
			start.z = simParams.nz;
		}
		if
		(
			(e.x >= start.x && e.x % 64 == 0 && e.y >= start.y && e.z >= start.z) &&
			(
			     e.x <= simParams.nx 
			  && e.y <= simParams.ny 
			  && e.z <= simParams.nz
			)
		) 
		{end = e;} 
		else
		{
			end.x = simParams.nx;
			end.y = simParams.ny;
			end.z = simParams.nz;
		}
		this->resizeTimings();
	}
	
	int3 urbTimer::getStart() const {return start;}
	int3 urbTimer::getEnd()   const	{return end;}
			
	void urbTimer::setStep(int3 const& t)
	{
		if(t.x >= 64 && t.x % 64 == 0 && t.y >- 1 && t.z >= 1) {step = t;}
		else
		{
			step.x = 64;
			step.y = 64;
			step.z = 1;
		}
		this->resizeTimings();
	}
	
	int3 urbTimer::getStep() const {return step;}
			
	void urbTimer::clearTimings() 
	{
		for(unsigned i = 0; i < timings.size(); i++)
		{
			timings[i] = 0.;
		}
	}
	
	float urbTimer::getTiming(int const& x, int const& y, int const& z) const
	{		
		int ndx = z * tDim.y * tDim.x + y * tDim.x + x;
		
		return (ndx < 0 || timings.size() <= (unsigned) ndx) ? -1. : timings[ndx] ;
	}
	
	float urbTimer::getTiming(float const& x, float const& y, float const& z) const
	{	
		int xndx = (x > end.x) ? end.x : (x - start.x) / step.x ;
		int yndx = (y > end.y) ? end.y : (y - start.y) / step.y ;
		int zndx = (z > end.z) ? end.z : (z - start.z) / step.z ;
			
		int ndx = zndx * tDim.y * tDim.x + yndx * tDim.x + xndx;
		
		return (ndx < 0 || timings.size() <= (unsigned) ndx) ? -1. : timings[ndx] ;
	}
						
	void urbTimer::runIterationTimings()
	{
		urbModule::quiet = true;
		simParams.diffusion_flag = false;
		urbModule::setMaxIterations(100);
		
		std::cout << "Running 100 iterations for each size." << std::endl;
	
		this->clearTimings();

		this->printInfo();

		for(float x = start.x; x <= end.x; x += step.x) // int version of setTiming broken...
		for(float y = start.y; y <= end.y; y += step.y)
		for(float z = start.z; z <= end.z; z += step.z)
		{	
			// HAHA!! Trickery!
			std::cout << std::fixed;
			std::cout.precision(0);
			std::cout << "Running dimension : " << x << "x" << y << "x" << z << std::flush;
			urbModule::setDimensions(x, y, z);
			QUIC::urbCUDA::solveUsingSOR_RB(this);
			
			// Register the appropriate time.
			double comput_time = urbModule::stpwtchs->comput->getElapsedTime();
			double denoms_time = urbModule::stpwtchs->denoms->getElapsedTime();
			
			this->setTiming(x, y, z, comput_time + denoms_time);
			std::cout.precision(5);
			std::cout << "\tcompute time : " << comput_time + denoms_time << std::flush;
			std::cout << "\titerations : " << iteration << std::endl;
			urbModule::reset();
		}
	}
	
	void urbTimer::outputTimings(std::string const& filename) const
	{
		std::ofstream output(filename.c_str());
		if(!output.is_open()) 
		{
			std::cerr << "Unable to open " << output << " for outputting urb timings.";
			return;
		}

		output << "start = {" << start.x << ", " << start.y << ", " << start.z << "}" << std::endl;
		output << "end   = {" << end.x   << ", " << end.y   << ", " << end.z   << "}" << std::endl;
		output << "step  = {" << step.x  << ", " << step.y  << ", " << step.z  << "}" << std::endl;		

		output << "max_iterations = " << simParams.max_iterations << std::endl;

		output << std::fixed;
		output.precision(6);
		output << "denoms time = " << urbModule::stpwtchs->denoms->getElapsedTime() << std::endl;

		for(float z = start.z; z <= end.z; z += step.z)
		{
			output.precision(0);
			output << "z = " << std::setw(6) << z;
			for(float x = start.x; x <= end.x; x += step.x) 
			{
				output << std::setw(10) << x;
			}
			output << " x" << std::endl;

			for(float y = start.y; y <= end.y; y += step.y)
			{
				output.precision(0);
				output << std::setw(10) << y;
				for(float x = start.x; x <= end.x; x += step.x)		
				{	
					output.precision(6);
					output << std::setw(10) << this->getTiming(x, y, z);
				}
				output << std::setw(10) << std::endl;
			}
			output << "y" << std::endl << std::endl;
		}
	}

	void urbTimer::printInfo() const
	{
		std::cout << "start = {" << start.x << ", " << start.y << ", " << start.z << "}" << std::endl;
		std::cout << "end   = {" << end.x   << ", " << end.y   << ", " << end.z   << "}" << std::endl;
		std::cout << "step  = {" << step.x  << ", " << step.y  << ", " << step.z  << "}" << std::endl;
	}

// Private
	void urbTimer::determineParameters(std::string& inp_dir)
	{
		standardFileParser* sfp = new standardFileParser();
		
		intElement
			ie_x_start = intElement("x_start"),
			ie_y_start = intElement("y_start"),
			ie_z_start = intElement("z_start"),
		
			ie_x_end   = intElement("x_end"),
			ie_y_end   = intElement("y_end"),
			ie_z_end   = intElement("z_end"),

			ie_x_step  = intElement("x_step"),
			ie_y_step  = intElement("y_step"),
			ie_z_step  = intElement("z_step");
		
		sfp->commit(ie_x_start);
		sfp->commit(ie_y_start);
		sfp->commit(ie_z_start);
		
		sfp->commit(ie_x_end);
		sfp->commit(ie_y_end);
		sfp->commit(ie_z_end);
		
		sfp->commit(ie_x_step);
		sfp->commit(ie_y_step);
		sfp->commit(ie_z_step);
		
		sfp->study("timing_range.txt");
		
		if
		(
			!sfp->recall(ie_x_start) ||
			!sfp->recall(ie_y_start) ||
			!sfp->recall(ie_z_start)
		) 
		{
			start.x = simParams.nx;
			start.y = simParams.ny;
			start.z = simParams.nz;
		}
		else
		{
			int x = ie_x_start.value;
			int y = ie_y_start.value;
			int z = ie_z_start.value;
	
			start.x = (0 < x && x < simParams.nx) ? x : simParams.nx ;
			start.y = (0 < y && y < simParams.ny) ? y : simParams.ny ;
			start.z = (0 < z && z < simParams.nz) ? z : simParams.nz ;
		}
		
		if
		(
			!sfp->recall(ie_x_end) ||
			!sfp->recall(ie_y_end) ||
			!sfp->recall(ie_z_end)
		)
		{
			end.x = simParams.nx;
			end.y = simParams.ny;
			end.z = simParams.nz;
		}
		else
		{
			int x = ie_x_end.value;
			int y = ie_y_end.value;
			int z = ie_z_end.value;
		
			end.x = (0 < x && x < simParams.nx) ? x : simParams.nx ;
			end.y = (0 < y && y < simParams.ny) ? y : simParams.ny ;
			end.z = (0 < z && z < simParams.nz) ? z : simParams.nz ;
		}
		
		if
		(
			!sfp->recall(ie_x_step) ||
			!sfp->recall(ie_y_step) ||
			!sfp->recall(ie_z_step)
		)
		{
			step.x = 64;
			step.y = 64;
			step.z = 1;
		}
		else
		{
			// x dimension limitation in CUDA.
			step.x = (ie_x_step.value % 64 == 0) ? ie_x_step.value : 64 ; 
			step.y = ie_y_step.value;
			step.z = ie_z_step.value;
		}
		std::cout << "done." << std::endl;

		delete sfp;

		this->resizeTimings();

		this->setRange(start, end);
		this->setStep(step);
	}
	
	void urbTimer::setTiming(int const& i, int const& j, int const& k, float const& time)
	{		
		// Doesn't seem to be working properly.
		int xndx = (i < 0) ? 0 : i ;
		int yndx = (j < 0) ? 0 : j ;
		int zndx = (k < 0) ? 0 : k ;
		
		if(tDim.x < xndx) xndx = tDim.x;
		if(tDim.y < yndx) yndx = tDim.y;
		if(tDim.z < zndx) zndx = tDim.z;
	
		timings[zndx*tDim.y*tDim.x + yndx*tDim.x + xndx] = time;
	}
	
	void urbTimer::setTiming(float const& x, float const& y, float const& z, float const& time)
	{
		int xndx = (x > end.x) ? end.x : (x - start.x) / step.x ;
		int yndx = (y > end.y) ? end.y : (y - start.y) / step.y ;
		int zndx = (z > end.z) ? end.z : (z - start.z) / step.z ;
		
		if(xndx < 0) xndx = 0;
		if(yndx < 0) yndx = 0;
		if(zndx < 0) zndx = 0;
	
		int ndx = zndx * tDim.y * tDim.x + yndx * tDim.x + xndx;
		timings[ndx] = time;
	}

	void urbTimer::resizeTimings()
	{
		//std::cout << "Resizing timings..." << std::flush;
		if(step.x < 1) step.x = 1;
		if(step.y < 1) step.y = 1;
		if(step.z < 1) step.z = 1;
	
		timings.clear();
		
		tDim.x = (end.x - start.x) / step.x + 1;
		tDim.y = (end.y - start.y) / step.y + 1;
		tDim.z = (end.z - start.z) / step.z + 1;
		
		//std::cout << tDim.x << std::endl;
		//std::cout << tDim.y << std::endl;
		//std::cout << tDim.z << std::endl;
		
		//std::cout << tDim.z*tDim.y*tDim.x + tDim.y*tDim.x + tDim.x << std::endl;
		
		timings.resize(tDim.z*tDim.y*tDim.x + tDim.y*tDim.x + tDim.x);
		//std::cout << "done." << std::endl;
	}
}

