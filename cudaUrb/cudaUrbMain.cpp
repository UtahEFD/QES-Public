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
#include "ESRIShapefile.h"

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

    
    // For now, process ESRIShapeFile here:
    ESRIShapefile *shpFile = nullptr;
    if (UID->simParams->shpFile != "") {
        shpFile = new ESRIShapefile( UID->simParams->shpFile,
                                     UID->simParams->shpBuildingLayerName );
        std::vector<float> shpDomainSize(2);
        shpFile->getLocalDomain( shpDomainSize );
        std::cout << "SHP Domain Size: " << shpDomainSize[0] << " X " << shpDomainSize[1] << std::endl;
    }
    

    // //////////////////////////////////////////
    //
    // Run the CUDA-URB Solver
    //
    // //////////////////////////////////////////
    Solver *solver, *solverC = nullptr;
    if (arguments.solveType == CPU_Type)
        solver = new CPUSolver(UID, DTEHF);
    else if (arguments.solveType == DYNAMIC_P)
        solver = new DynamicParallelism(UID, DTEHF);
    else
    {
        std::cerr << "Error: invalid solve type\n";
        exit(EXIT_FAILURE);
    }

    //check for comparison
    if (arguments.compareType)
    {
        if (arguments.compareType == CPU_Type)
            solverC = new CPUSolver(UID, DTEHF);
        else if (arguments.compareType == DYNAMIC_P)
            solverC = new DynamicParallelism(UID, DTEHF);
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

    // Prepare the NetCDF Data Structure for outputting simulation results.
    NetCDFData* netcdfDat = nullptr, *netcdfCompare = nullptr;
    netcdfDat = new NetCDFData();


    if (!arguments.solveWind) {

        solver->outputNetCDF( netcdfDat );
	    solver->outputDataFile ();

        if (solverC != nullptr)
        {
            netcdfCompare = new NetCDFData();
            solverC->outputNetCDF( netcdfCompare);
            if (!netcdfDat->outputCellResultsDifference(netcdfCompare, arguments.netCDFFile))
            {   
                cerr << "ERROR: output is broken\n";
                return -1;
            }
        }
        if (!netcdfDat->outputCellFaceResults(arguments.netCDFFile))
        {
            cerr << "ERROR: output is broken\n";
            return -1;
        }
    }


    if (arguments.iCellOut != "")
    {
        if (!netcdfDat->outputICellFlags(arguments.iCellOut))
        {
            cerr << "ERROR: iCell is broken\n";
            return -2;
        }
        if (DTEHF)
            if (!netcdfDat->outputCutCellFlags(arguments.iCellOut))
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
