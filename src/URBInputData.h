#pragma once

#include "util/ParseInterface.h"

#include "SimulationParameters.h"
#include "FileOptions.h"
#include "MetParams.h"
#include "Buildings.h"

class URBInputData : public ParseInterface
{
public:
	SimulationParameters* simParams;
	FileOptions* fileOptions;
	MetParams* metParams;
	Buildings* buildings;


	URBInputData()
	{
	fileOptions = 0;
	metParams = 0;
	buildings = 0;
	}

	virtual void parseValues()
	{
	parseElement<SimulationParameters>(true, simParams, "simulationParameters");
	parseElement<FileOptions>(false, fileOptions, "fileOptions");
	parseElement<MetParams>(false, metParams, "metParams");
	parseElement<Buildings>(false, buildings, "buildings");
	}

    	/**
	 * This function takes in an URBInputData variable and uses it
	 * as the base to parse the ptree
	 * @param UID the object that will serve as the base level of the xml parser
	 */
    void parseTree(pt::ptree t) { //  URBInputData*& UID) {
        // root = new URBInputData();
        setTree(t);
        setParents("root");
        parseValues();
    }
    

};
