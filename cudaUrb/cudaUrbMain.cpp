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
#include "NetCDFData.h"
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

    DTEHeightField* DTEHF = 0;
    if (arguments.demFile != "") {
        DTEHF = new DTEHeightField(arguments.demFile, (*(UID->simParams->grid))[0], (*(UID->simParams->grid))[1] );
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

    // Files was successfully read
    std::cout << "Data Was Read\n";

    // //////////////////////////////////////////
    //
    // Run the CUDA-URB Solver
    //
    // //////////////////////////////////////////
    Solver* solver;
    if (arguments.solveType == CPU_Type)
        solver = new CPUSolver(UID, DTEHF);
    else if (arguments.solveType == DYNAMIC_P)
        solver = new DynamicParallelism(UID, DTEHF);
    else
    {
        std::cerr << "Error: invalid solve type\n";
        exit(EXIT_FAILURE);
    }


    // Run urb simulation code
    solver->solve( !arguments.solveWind);


    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////

    // Prepare the NetCDF Data Structure for outputting simulation results.
    NetCDFData* netcdfDat = 0;
    netcdfDat = new NetCDFData();


    if (!arguments.solveWind) {

        solver->outputNetCDF( netcdfDat );
	solver->outputDataFile ();

        if (!netcdfDat->outputCellFaceResults(arguments.netCDFFile))
        {
            cerr << "ERROR: output is broken\n";
            return -1;
        }
    }


    if (arguments.iCellOut)
    {
        if (!netcdfDat->outputICellFlags("iCellValues.nc"))
        {
            cerr << "ERROR: iCell is broken\n";
            return -2;
        }
        if (DTEHF)
            if (!netcdfDat->outputCutCellFlags("cutCellFlags.nc"))
            {
                cerr << "ERROR: cutCell is broken\n";
                return -3;
            }
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

//	try
//	{
//		ParseInterface::parseTree(tree, xmlRoot);
//	}
//	catch (ParseException& p )
//	{
//		std::cerr  << "ERROR: " << p.what() << std::endl;
//		xmlRoot = 0;
//	}

	return xmlRoot;
}
