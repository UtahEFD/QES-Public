#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <limits>

#include "WINDSGeneralData.h"
#include "TURBGeneralData.h"
#include "TURBWall.h"



class TURBWallTerrain : public TURBWall
{
protected:
public:

    TURBWallTerrain()
    {}
    ~TURBWallTerrain()
    {}

    /**
     * @brief
     *
     * This function takes in the icellflags set by setCellsFlag
     * function for stair-step method and sets related coefficients to
     * zero to define solid walls. It also creates vectors of indices
     * of the cells that have wall to right/left, wall above/bellow
     * and wall in front/back
     *
     */
    void defineWalls(WINDSGeneralData*,TURBGeneralData*);
    void setWallsBC(WINDSGeneralData*,TURBGeneralData*);

private:
    const int icellflag_terrain = 2;
    const int icellflag_cutcell = 8;

    const int iturbflag_stairstep = 2;
    const int iturbflag_cutcell = 3;

    bool use_cutcell = false;

};
