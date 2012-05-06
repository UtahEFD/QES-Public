/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Driver / Stub for urbTimer class.
*/

#include <iostream>
 
#include "urbParser.h"
#include "urbSetup.h"
#include "urbTimer.h"

#include "../util/directory.h"

int main(int argc, char* argv[]) 
{
	std::cout << std::endl << "###--- Extensive timings for urbCUDA on device ---###" << std::endl;
	
	std::string inp_dir;
	std::string out_dir; // Puts output in local directory.
	
	getDirectories(argc, argv, inp_dir, out_dir);

	std::string output_file = out_dir + "cuda.tms";
	
	QUIC::urbTimer* ut = new QUIC::urbTimer();
	QUIC::urbParser::parse(ut, inp_dir);
	
	ut->setEpsilon(0.);
	ut->setMaxIterations(100);
	
	//QUIC::urbSetup::usingFortran(ut);
	
	ut->determineParameters(inp_dir);
	ut->runIterationTimings();
	ut->outputTimings(output_file);
	
	delete ut;
	
	return 0;
}

