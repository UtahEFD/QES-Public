#include "ParseInterface.h"
#include "Root.h"
#include "X.h"
#include "A.h"
#include "B.h"
#include "C.h"
#include "D.h"
#include "P.h"

#include <string>
#include <iostream>
#include <iomanip>

#include <vector>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

int main(int argc, char *argv[])
{

	std::string fileName = argv[1];
	pt::ptree tree;

	pt::read_xml(fileName, tree);

	ParseInterface structureParsed(tree);

	structureParsed.parseValues();

	std::cout << "X" << std::endl << structureParsed.root->xVar->x << std::endl;
	std::cout << "A" << std::endl;
	for (unsigned int i = 0; i < structureParsed.root->xVar->aVar->vals.size(); i++)
		std::cout << structureParsed.root->xVar->aVar->vals[i] << std::endl;
	std::cout << "B" << std::endl;
	std::cout << structureParsed.root->xVar->bVar->numAs << std::endl;
	for (unsigned int i = 0; i < structureParsed.root->xVar->bVar->aVals.size(); i++)
	{
		std::cout << "A" << std::endl;
		for (unsigned int j = 0; j < (structureParsed.root->xVar->bVar->aVals[i])->vals.size(); j++)
			std::cout << (structureParsed.root->xVar->bVar->aVals[i])->vals[j] << std::endl;
	}
	std::cout << "C" << std::endl;
	std::cout << structureParsed.root->xVar->cVar->x << std::endl << structureParsed.root->xVar->cVar->y << std::endl;
	structureParsed.root->xVar->pVar->output();

	std::cout << "D" << std::endl;
	for (unsigned int i = 0; i < structureParsed.root->xVar->dVar->pVars.size(); i++)
		structureParsed.root->xVar->dVar->pVars[i]->output();


}