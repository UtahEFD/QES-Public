#include <iostream>

#include "ParseException.h"
#include "ParseInterface.h"
#include "URBInputData.h"
#include "handleURBArgs.h"
#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"
#include "NetCDFData.h"
#include "DTEHeightField.h"

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>


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

    // Use Pete's arg parser for command line stuff...
    URBArgs arguments;
    arguments.processArguments(argc, argv);

    NetCDFData* netcdfDat;
    netcdfDat = new NetCDFData();

    std::cout << "cudaUrb " << "0.8.0" << std::endl;

    // read input files  -- some test XML, netcdf.... for now...
    URBInputData* UID;

    std::cout << "parsing xml...\n";

	UID = parseXMLTree(arguments.quicFile);

	std::cout << "parsing complete!\n";

	if ( UID )
	{
	    DTEHeightField* DTEHF = 0;
	    if (arguments.demFile != "")
	    {
	    	DTEHF = new DTEHeightField(arguments.demFile, (*(UID->simParams->grid))[0], (*(UID->simParams->grid))[1] );
	    }

		if (DTEHF)
		{
			std::cout << "Forming triangle mesh...\n";
			DTEHF->setDomain(UID->simParams->domain, UID->simParams->grid);
			std::cout << "Mesh complete\n";
		}

		if (arguments.terrainOut)
		{
			if (DTEHF)
			{
				std::cout << "Creating terrain OBJ....\n";
				DTEHF->outputOBJ("terrain.obj");
				std::cout << "OBJ created....\n";
			}
			else
			{
				std::cerr << "Error: No dem file specified as input\n";
				return -1;				
			}
		}

		std::cout << "Data Was Read\n";
		//File was successfully read

		
		Solver* solver;

		if (arguments.solveType == CPU_Type)
			solver = new CPUSolver(UID, DTEHF);
		else if (arguments.solveType == DYNAMIC_P)
			solver = new DynamicParallelism(UID, DTEHF);

		else
		{
			std::cerr << "Error: invalid solve type\n";
			return -1;
		}
    
    	// Run Simulation code
		solver->solve(netcdfDat, !arguments.solveWind);


   		// output netcdf test file
   		if (!arguments.solveWind)
	   		if (!netcdfDat->outputCellFaceResults(arguments.netCDFFile))
	   		{
	    		cerr << "ERROR: output is broken\n";
	    		return -1;
	   		}

   		if (arguments.iCellOut)
	   		if (!netcdfDat->outputICellFlags("iCellValues.nc"))
	   		{
	    		cerr << "ERROR: iCell is broken\n";
	    		return -2;
	   		}
	}

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
		std::cerr << "Error reading tree\n";
		return (URBInputData*)0;
	}

	URBInputData* xmlRoot;
	try 
	{
		ParseInterface::parseTree(tree, xmlRoot);
	}
	catch (ParseException& p )
	{
		std::cerr  << "ERROR: " << p.what() << std::endl;
		xmlRoot = 0;
	}
	return xmlRoot;
}