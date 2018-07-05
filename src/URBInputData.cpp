#include "URBInputData.h"


URBInputData::URBInputData()
{
	fileOptions = 0;
	metParams = 0;
	buildings = 0;
	terrain = 0;
}

void URBInputData::parseValues()
{
	parseElement<SimulationParameters>(true, simParams, "simulationParameters");
	parseElement<FileOptions>(false, fileOptions, "fileOptions");
	parseElement<MetParams>(false, metParams, "metParams");
	parseElement<Buildings>(false, buildings, "buildings");
	parseElement<Terrain>(false, terrain, "terrain");
}