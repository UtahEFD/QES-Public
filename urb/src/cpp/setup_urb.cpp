/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Driver / Stub for running / testing setup for Fortran and C++.
*/

#include <iostream>

#include "urbCUDA.h"
#include "urbSetup.h"
#include "urbParser.h"

#include "../util/matrixIO.h"
#include "../util/directory.h" 

int main(int argc, char* argv[]) 
{
	std::string inp_dir;
	std::string out_dir;
	
	getDirectories(argc, argv, inp_dir, out_dir);

	QUIC::urbModule* um_cpp = new QUIC::urbModule(); um_cpp->beQuiet(true);
	QUIC::urbModule* um_frt = new QUIC::urbModule(); um_frt->beQuiet(true);

	um_cpp->setName("CPP");
	um_frt->setName("Fortran");
	
	QUIC::urbParser::parse(um_cpp, inp_dir);
	QUIC::urbSetup::usingCPP(um_cpp);
	//QUIC::urbCUDA::solveUsingSOR_RB(um_cpp);

  QUIC::urbParser::parse(um_frt, inp_dir);
	QUIC::urbSetup::usingFortran(um_frt);
	//QUIC::urbCUDA::solveUsingSOR_RB(um_frt);

	std::cout << std::endl << "CPP";
	QUIC::urbParser::printSetupTimes(um_cpp);
	std::cout << std::endl << "Fortran";
	QUIC::urbParser::printSetupTimes(um_frt);	

/*	
	float* frt_p1 = um_frt->getSolution();
	float* cpp_p1 = um_cpp->getSolution();
	
	float absolute_difference = matricesAbsoluteDifference
															(
																frt_p1, 
																um_frt->getNX(), 
																um_frt->getNX(), 
																um_frt->getNX(), 
																cpp_p1,
																um_cpp->getNX(), 
																um_cpp->getNY(),
																um_cpp->getNZ()
															);

	std::cout << "When initializing with Fortran versus CPP, " << std::endl;
	std::cout << "  Absolute Difference = " << absolute_difference << std::endl;

	int loc[3] = {0,0,0};
	float max_difference = matricesMaxDifference
												 (
													 frt_p1, 
													 um_frt->getNX(), 
													 um_frt->getNX(), 
													 um_frt->getNX(), 
													 cpp_p1,
													 um_cpp->getNX(), 
													 um_cpp->getNY(),
													 um_cpp->getNZ(),
													 loc
												 );
	std::cout << "  Max Difference = " << max_difference << " at " << std::flush;
	std::cout << "(" << loc[0] << ", " << loc[1] << ", " << loc[2] << ")" << std::endl;
*/
	
	delete um_cpp;
	delete um_frt;
	
	return 0;
}

