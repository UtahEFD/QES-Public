#pragma once

/*
 * This function contains variables that define information
 * necessary for running the simulation.
 */

#include "util/ParseInterface.h"
#include "Vector3.h"

class SimulationParameters : public ParseInterface
{
private:



public:

	Vector3<int>* domain;
	Vector3<float>* grid;
	int verticalStretching;
	int totalTimeIncrements;
	int rooftopFlag;
	int upwindCavityFlag;
	int streetCanyonFlag;
	int streetIntersectionFlag;
	int wakeFlag;
	int sidewallFlag;
	int maxIterations;
	int residualReduction;
	float domainRotation;
	float UTMx;
	float UTMy;
	int UTMZone;
	int UTMZoneLetter;
	int meshTypeFlag;

	SimulationParameters()
	{
		UTMx = 0.0;
		UTMy = 0.0;
		UTMZone = 0;
		UTMZoneLetter = 0;
	}

	virtual void parseValues()
	{
		parseElement< Vector3<int> >(true, domain, "domain");
		parseElement< Vector3<float> >(true, grid, "cellSize");
		parsePrimitive<int>(true, verticalStretching, "verticalStretching");
		parsePrimitive<int>(true, totalTimeIncrements, "totalTimeIncrements");
		parsePrimitive<int>(true, rooftopFlag, "rooftopFlag");
		parsePrimitive<int>(true, upwindCavityFlag, "upwindCavityFlag");
		parsePrimitive<int>(true, streetCanyonFlag, "streetCanyonFlag");
		parsePrimitive<int>(true, streetIntersectionFlag, "streetIntersectionFlag");
		parsePrimitive<int>(true, wakeFlag, "wakeFlag");
		parsePrimitive<int>(true, sidewallFlag, "sidewallFlag");
		parsePrimitive<int>(true, maxIterations, "maxIterations");
		parsePrimitive<int>(true, residualReduction, "residualReduction");
		parsePrimitive<int>(true, meshTypeFlag, "meshTypeFlag");
		parsePrimitive<float>(true, domainRotation, "domainRotation");
		parsePrimitive<float>(false, UTMx, "UTMx");
		parsePrimitive<float>(false, UTMy, "UTMy");
		parsePrimitive<int>(false, UTMZone, "UTMZone");
		parsePrimitive<int>(false, UTMZoneLetter, "UTMZoneLetter");
	}

};
