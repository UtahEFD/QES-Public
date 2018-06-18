#include "URBInputData.h"

void URBInputData::parseValues()
{
	parseElement<SimulationParameters>(true, simParams, "simulationParameters");
	parseElement<FileOptions>(false, fileOptions, "fileOptions");
	parseElement<MetParams>(false, metParams, "metParams");
	parseElement<Buildings>(false, buildings, "buildings");
}