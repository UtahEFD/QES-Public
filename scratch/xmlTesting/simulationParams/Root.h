#pragma once

#include "ParseInterface.h"
#include "SimulationParameters.h"

class Root : public ParseInterface
{
public:
	SimulationParameters* simParams;

	void parseValues()
	{
		parseElement<SimulationParameters>(false, simParams, "simulationParameters");
	}
};