#pragma once

#include "ParseInterface.h"
#include "Vector3.h"

class SimulationParameters : public ParseInterface
{
private:



public:

	Vector3<int>* domain;
	Vector3<float>* grid;
	int verticalStretching;
	int totalTimeIncrements;
	int UTCConversion;
	float Epoch;
	int rooftopFlag;
	int upwindCavityFlag;
	int streetCanyonFlag;
	int streetIntersectionFlag;
	int wakeFlag;
	int sidewallFlag;
	int maxIterations;
	int residualReduction;
	int useDiffusion;
	float domainRotation;
	int UTMX;
	int UTMY;
	int UTMZone;
	int UTMZoneLetter;
	int quicCDFFlag;
	int explosiveDamageFlag;
	int buildingArrayFlag;
	
	SimulationParameters()
	{
		UTMX = 0;
		UTMY = 0;
		UTMZone = 0;
		UTMZoneLetter = 0;
		quicCDFFlag = 0;
		explosiveDamageFlag = 0;
		buildingArrayFlag = 0;
	}

	virtual void parseValues()
	{
		parseElement< Vector3<int> >(true, domain, "domain");
		parseElement< Vector3<float> >(true, grid, "cellSize");
		parsePrimative<int>(true, verticalStretching, "verticalStretching");
		parsePrimative<int>(true, totalTimeIncrements, "totalTimeIncrements");
		parsePrimative<int>(true, UTCConversion, "UTCConversion");
		parsePrimative<float>(true, Epoch, "Epoch");
		parsePrimative<int>(true, rooftopFlag, "rooftopFlag");
		parsePrimative<int>(true, upwindCavityFlag, "upwindCavityFlag");
		parsePrimative<int>(true, streetCanyonFlag, "streetCanyonFlag");
		parsePrimative<int>(true, streetIntersectionFlag, "streetIntersectionFlag");
		parsePrimative<int>(true, wakeFlag, "wakeFlag");
		parsePrimative<int>(true, sidewallFlag, "sidewallFlag");
		parsePrimative<int>(true, maxIterations, "maxIterations");
		parsePrimative<int>(true, residualReduction, "residualReduction");
		parsePrimative<int>(true, useDiffusion, "useDiffusion");
		parsePrimative<float>(true, domainRotation, "domainRotation");
		parsePrimative<int>(false, UTMX, "UTMX");
		parsePrimative<int>(false, UTMY, "UTMY");
		parsePrimative<int>(false, UTMZone, "UTMZone");
		parsePrimative<int>(false, UTMZoneLetter, "UTMZoneLetter");
		parsePrimative<int>(false, quicCDFFlag, "quicCDFFlag");
		parsePrimative<int>(false, explosiveDamageFlag, "explosiveDamageFlag");
		parsePrimative<int>(false, buildingArrayFlag, "buildingArrayFlag");
	}
};