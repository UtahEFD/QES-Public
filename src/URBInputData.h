#pragma once

#include "ParseInterface.h"
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
};