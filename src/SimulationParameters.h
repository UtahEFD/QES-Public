#pragma once

#include "ParseInterface.h"
#include "Vector3.h"

<<<<<<< HEAD
=======
/*
 *Placeholder class for parsed simulation parameters info in the xml
 */
>>>>>>> origin/doxygenAdd
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
<<<<<<< HEAD
=======
	int useDiffusion;
>>>>>>> origin/doxygenAdd
	float domainRotation;
	int UTMX;
	int UTMY;
	int UTMZone;
	int UTMZoneLetter;
<<<<<<< HEAD
=======
	int quicCDFFlag;
	int explosiveDamageFlag;
	int buildingArrayFlag;
>>>>>>> origin/doxygenAdd
	std::vector<float> dzArray;
	
	SimulationParameters()
	{
		UTMX = 0;
		UTMY = 0;
		UTMZone = 0;
		UTMZoneLetter = 0;
<<<<<<< HEAD
=======
		quicCDFFlag = 0;
		explosiveDamageFlag = 0;
		buildingArrayFlag = 0;
>>>>>>> origin/doxygenAdd
	}

	virtual void parseValues()
	{
		parseElement< Vector3<int> >(true, domain, "domain");
		parseElement< Vector3<float> >(true, grid, "cellSize");
		parsePrimitive<int>(true, verticalStretching, "verticalStretching");
		parsePrimitive<int>(true, totalTimeIncrements, "totalTimeIncrements");
		parsePrimitive<int>(true, UTCConversion, "UTCConversion");
		parsePrimitive<float>(true, Epoch, "Epoch");
		parsePrimitive<int>(true, rooftopFlag, "rooftopFlag");
		parsePrimitive<int>(true, upwindCavityFlag, "upwindCavityFlag");
		parsePrimitive<int>(true, streetCanyonFlag, "streetCanyonFlag");
		parsePrimitive<int>(true, streetIntersectionFlag, "streetIntersectionFlag");
		parsePrimitive<int>(true, wakeFlag, "wakeFlag");
		parsePrimitive<int>(true, sidewallFlag, "sidewallFlag");
		parsePrimitive<int>(true, maxIterations, "maxIterations");
		parsePrimitive<int>(true, residualReduction, "residualReduction");
<<<<<<< HEAD
=======
		parsePrimitive<int>(true, useDiffusion, "useDiffusion");
>>>>>>> origin/doxygenAdd
		parsePrimitive<float>(true, domainRotation, "domainRotation");
		parsePrimitive<int>(false, UTMX, "UTMX");
		parsePrimitive<int>(false, UTMY, "UTMY");
		parsePrimitive<int>(false, UTMZone, "UTMZone");
		parsePrimitive<int>(false, UTMZoneLetter, "UTMZoneLetter");
<<<<<<< HEAD
		parseMultiPrimitives<float>(false, dzArray, "dz_array");
	}

};
=======
		parsePrimitive<int>(false, quicCDFFlag, "quicCDFFlag");
		parsePrimitive<int>(false, explosiveDamageFlag, "explosiveDamageFlag");
		parsePrimitive<int>(false, buildingArrayFlag, "buildingArrayFlag");
		parseMultiPrimitives<float>(false, dzArray, "dz_array");
	}

};
>>>>>>> origin/doxygenAdd
