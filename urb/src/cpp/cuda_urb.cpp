/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Driver / Stub for running QUICurb with C++ / CUDA.
*/

#include <iostream>
 
#include "urbCUDA.h"
#include "urbModule.h"
#include "urbParser.h"
#include "urbSetup.h"

#include "../util/directory.h"

int main(int argc, char* argv[]) 
{
	std::cout << std::endl << "###--- sor3d on device ---###" << std::endl;
	
	std::string inp_dir;
	std::string out_dir;

	getDirectories(argc, argv, inp_dir, out_dir);

	QUIC::urbModule* um = new QUIC::urbModule();
        um->beQuiet(false);
  
	QUIC::urbParser::parse(um, inp_dir);
	QUIC::urbSetup::setup(um);

	um->sanityCheck();

	for(int i = 0; i < 1; i++)
	{
		um->reset();
		
		QUIC::urbCUDA::solveUsingSOR_RB(um);
		//QUIC::urbParser::printIterTimes(um);
				
		um->printLastError();
	}
	

  QUIC::urbParser::printInfo(um);
  
  // Output flags for dumping.
  um->output_celltypes = true;
  um->output_boundaries = true;
  um->output_divergence = true;
  um->output_denominators = true;
//  um->output_lagrangians = true;
  um->output_velocities = true;

	QUIC::urbParser::dump(um, out_dir);
//	QUIC::urbParser::generatePlumeInput(um, out_dir);

/*
	QUIC::angle w      = QUIC::angle(  0., QUIC::DEG, QUIC::MET); 
	QUIC::angle a_end  = QUIC::angle(359., QUIC::DEG, QUIC::MET);
	QUIC::angle a_incr = QUIC::angle( 20., QUIC::DEG, QUIC::MET);

	for(; w <= a_end; w += a_incr)
	{
		std::cout << std::fixed << std::endl << std::endl;
		std::cout.precision(2);
		std::cout << "Setting up and solving for wind angle : " << w << std::endl;
		um->setWindAngle(w);
		QUIC::urbSetup::usingCPP(um);
		QUIC::urbCUDA::solveUsingSOR_RB(um);
		QUIC::urbParser::printInfo(um);
	}
*/
	delete um;

	return 0;
}

