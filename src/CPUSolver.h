#pragma once

#include "URBInputData.h"
#include "Solver.h"
#include <chrono>

class CPUSolver : public Solver
{
public:
	CPUSolver(URBInputData* UID)
		: Solver(UID)
		{

		}

	virtual void solve();

};