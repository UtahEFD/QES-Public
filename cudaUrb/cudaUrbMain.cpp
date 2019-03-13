#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "URBInputData.h"
#include "handleURBArgs.h"
#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"
#include "Output.hpp"
#include "DTEHeightField.h"

namespace pt = boost::property_tree;

/**
 * This function takes in a filename and attempts to open and parse it.
 * If the file can't be opened or parsed properly it throws an exception,
 * if the file is missing necessary data, an error will be thrown detailing
 * what data and where in the xml the data is missing. If the tree can't be
 * parsed, the Root* value returned is 0, which will register as false if tested.
 * @param fileName the path/name of the file to be opened, must be an xml
 * @return A pointer to a root that is filled with data parsed from the tree
 */
URBInputData* parseXMLTree(const std::string fileName);

int main(int argc, char *argv[])
{
    std::string Revision = "0";
    // CUDA-Urb - Version output information
    std::cout << "cudaUrb " << "0.8.0" << std::endl;

    // ///////////////////////////////////
    // Parse Command Line arguments
    // ///////////////////////////////////

    // Command line arguments are processed in a uniform manner using
    // cross-platform code.  Check the URBArgs class for details on
    // how to extend the arguments.
    URBArgs arguments;
    arguments.processArguments(argc, argv);

    // ///////////////////////////////////
    // Read and Process any Input for the system
    // ///////////////////////////////////

    // Parse the base XML QUIC file -- contains simulation parameters
    URBInputData* UID = parseXMLTree(arguments.quicFile);
    if ( !UID ) {
        std::cerr << "QUIC Input file: " << arguments.quicFile << " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }

    //if the commandline dem file is blank, and a file was specified in the xml,
    //use the dem file from the xml
    std::string demFile = "";
    if (arguments.demFile != "")
        demFile = arguments.demFile;
    else if (UID->simParams && UID->simParams->demFile != "")
        demFile = UID->simParams->demFile;


    DTEHeightField* DTEHF = 0;
    if (demFile != "") {
        DTEHF = new DTEHeightField(demFile, (*(UID->simParams->grid))[0], (*(UID->simParams->grid))[1] );
    }

    if (DTEHF) {
        std::cout << "Forming triangle mesh...\n";
        DTEHF->setDomain(UID->simParams->domain, UID->simParams->grid);
        std::cout << "Mesh complete\n";
    }

    if (arguments.terrainOut) {
        if (DTEHF) {
            std::cout << "Creating terrain OBJ....\n";
            DTEHF->outputOBJ("terrain.obj");
            std::cout << "OBJ created....\n";
        }
        else {
            std::cerr << "Error: No dem file specified as input\n";
            return -1;
        }
    }
    
    // Files was successfully read, so create instance of output class
    Output* output = nullptr;
    if (UID->fileOptions->outputFlag==1) {
        output = new Output(arguments.netCDFFile);
    }

    // //////////////////////////////////////////
    //
    // Run the CUDA-URB Solver
    //
    // //////////////////////////////////////////
    Solver *solver, *solverC = nullptr;
    if (arguments.solveType == CPU_Type)
        solver = new CPUSolver(UID, DTEHF, output);
    else if (arguments.solveType == DYNAMIC_P)
        solver = new DynamicParallelism(UID, DTEHF, output);
    else
    {
        std::cerr << "Error: invalid solve type\n";
        exit(EXIT_FAILURE);
    }

    //check for comparison
    if (arguments.compareType)
    {
        if (arguments.compareType == CPU_Type)
            solverC = new CPUSolver(UID, DTEHF, output);
        else if (arguments.compareType == DYNAMIC_P)
            solverC = new DynamicParallelism(UID, DTEHF, output);
        else
        {
            std::cerr << "Error: invalid comparison type\n";
            exit(EXIT_FAILURE);
        }
    }
    
    //close the scanner
    if (DTEHF)
        DTEHF->closeScanner();
    
    // Run urb simulation code
    solver->solve( !arguments.solveWind);

    std::cout << "Solver done!\n";

    if (solverC != nullptr)
    {
        std::cout << "Running comparson type...\n";
        solverC->solve(!arguments.solveWind);
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    if (output != nullptr) {
        std::cout << "Saving data!"<<std::endl;
        solver->save(output);
    }
    
    exit(EXIT_SUCCESS);
}

URBInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (URBInputData*)0;
	}

	URBInputData* xmlRoot = new URBInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}
