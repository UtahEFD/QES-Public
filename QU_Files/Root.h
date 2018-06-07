#pragma once

#include "ParseInterface.h"
#include "SimulationParameters.h"
#include "FileOptions.h"
#include "MetParams.h"
#include "Buildings.h"

class Root : public ParseInterface
{
public:
	SimulationParameters* simParams;
	FileOptions* fileOptions;
	MetParams* metParams;
	Buildings* buildings;

	void parseValues()
	{
		parseElement<SimulationParameters>(false, simParams, "simulationParameters");
		parseElement<SimulationParameters>(false, fileOptions, "fileOptions");
		parseElement<SimulationParameters>(false, metParams, "metParams");
		parseElement<SimulationParameters>(false, buildings, "buildings");
	}
};