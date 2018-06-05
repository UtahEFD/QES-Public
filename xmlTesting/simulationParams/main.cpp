#include "ParseException.h"
#include "ParseInterface.h"
#include "Root.h"

#include <string>
#include <iostream>
#include <iomanip>

#include <vector>

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
Root* parseXMLTree(const std::string fileName);

int main(int argc, char *argv[])
{

	std::string fileName = argv[1];
	Root* root;

	root = parseXMLTree(fileName);
//	if ( root )
//	{

//	}

}

Root* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try 
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree\n";
		return (Root*)0;
	}

	Root* xmlRoot;
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