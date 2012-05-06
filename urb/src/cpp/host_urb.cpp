/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Driver / Stub for running QUICurb with host C++.
*/

#include <iostream>
 
#include "urbHost.h"
#include "urbModule.h"
#include "urbParser.h"
#include "urbSetup.h"

#include "../util/directory.h" 

int main(int argc, char* argv[]) 
{
	std::cout << std::endl << "###--- sor3d on host (C++) ---###" << std::endl;
	
	std::string inp_dir;
	std::string out_dir;

	getDirectories(argc, argv, inp_dir, out_dir);

	QUIC::urbModule* um = new QUIC::urbModule();
	um->beQuiet(false);
	
	QUIC::urbParser::parse(um, inp_dir);
	QUIC::urbSetup::setup(um);

	QUIC::urbHost::solveUsingSOR_RB(um);

  QUIC::urbParser::printInfo(um);
//	QUIC::urbParser::dump(um, out_dir);

	delete um;
	
	return 0;
}

