#pragma once

#include "ParseInterface.h"
#include "Vector3.h"

class SimulationParameters : public ParseInterface
{
private:
	Vector3<int>* domain;
	Vector3<float>* grid;
	int utc, upwind, streetCanyon, streetIntersection, wake, sidewall, iterationMax, residual, diffusion, diffusionIterations;
	std::string rooftop;
public:
	virtual void parseValues()
	{
		parseElement< Vector3<int> >(true, domain, "domain");
		parseElement< Vector3<float> >(true, grid, "grid");
		parsePrimative<int>(true, utc, "utc");
		parsePrimative<std::string>(true, rooftop, "rooftop");
		parsePrimative<int>(true, upwind, "upwind");
		parsePrimative<int>(true, streetCanyon, "streetCanyon");
		parsePrimative<int>(true, streetIntersection, "streetIntersection");
		parsePrimative<int>(true, wake, "wake");
		parsePrimative<int>(true, sidewall, "sidewall");
		parsePrimative<int>(true, iterationMax, "iterationMax");
		parsePrimative<int>(true, residual, "residual");
		parsePrimative<int>(true, diffusion, "diffusion");
		parsePrimative<int>(true, diffusionIterations, "diffusionIterations");
	}
};