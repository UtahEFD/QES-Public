#include <iostream>

#include "ParseException.h"
#include "ParseInterface.h"
#include "URBInputData.h"

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

    std::cout << "cudaUrb " << "0.8.0" << std::endl;

    // read input files  -- some test XML, netcdf.... for now...
    URBInputData* UID;

	UID = parseXMLTree("QU_Files/QU_inner.xml");
	if ( UID )
	{
		std::cout << "FileWasRead\n";
		//File was successfully read
		int x = 5; //Dummy lines
		x++;       //for compile
	
    

    	// Run Simulation code



   		// output netcdf test file
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