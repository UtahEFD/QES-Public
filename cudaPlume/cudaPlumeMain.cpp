#include <iostream>
#include <netcdf>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "Plume.hpp"
#include "Args.hpp"
#include "Urb.hpp"
#include "Turb.hpp"
#include "Input.hpp"
#include "Output.hpp"
#include "Eulerian.h"
#include "Dispersion.h"
#include "PlumeInputData.hpp"

#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <ctime>

using namespace netCDF;
using namespace netCDF::exceptions;

PlumeInputData* parseXMLTree(const std::string fileName);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    // set up time information
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // print a nice little welcome message
    std::cout << std::endl;
    std::cout<<"##############################################################"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"#                   Welcome to CUDA-PLUME                    #"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    
    // parse command line arguments
    Args arguments;
    arguments.processArguments(argc, argv);
    
    // parse xml settings
    PlumeInputData* PID = parseXMLTree(arguments.quicFile);
    if ( !PID ) {
        std::cerr << "QUIC-Plume input file: " << arguments.quicFile << " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }
        
    // Create instance of cudaUrb input class
    Input* inputUrb = new Input(arguments.inputFileUrb);
    
    // Create instance of cudaTurb input class
    Input* inputTurb = new Input(arguments.inputFileTurb);

    // Create instance of output class
    Output* output = new Output(arguments.outputFile);
    
    // Create instance of cudaUrb class
    Urb* urb = new Urb(inputUrb);
    
    // Create instance of cudaTurb class
    Turb* turb = new Turb(inputTurb);
    
    // Create instance of Eulerian class
    Eulerian* eul = new Eulerian(urb,turb);
    
    // Create instance of Dispersion class
    Dispersion* dis = new Dispersion(urb,turb,PID);
    
    // Create instance of Plume model class
    Plume* plume = new Plume(urb,dis,PID);
    
    // Run plume advection model
    plume->run(urb,turb,eul,dis,PID,output);
    
    // compute run time information
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    std::cout<<"[CUDA-Plume] \t Finished in "<<elapsed<<" seconds!"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    
    exit(EXIT_SUCCESS);
  
}

PlumeInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (PlumeInputData*)0;
	}

	PlumeInputData* xmlRoot = new PlumeInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}
 
