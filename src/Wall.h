#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <limits>

// may need to forward reference this???
class URBGeneralData;

using namespace std;

class Wall
{

protected:


public:

    Wall()
    {

    }
    ~Wall()
    {

    }

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
    void defineWalls(URBGeneralData *UGD);


    /**
    * @brief
    *
    * This function takes in vectores of indices of the cells that have wall to right/left,
    * wall above/bellow and wall in front/back and applies the log law boundary condition fix
    * to the cells near Walls (based on Kevin Briggs master's thesis (2015))
    *
    */
    void wallLogBC (URBGeneralData *UGD);


    /**
    * @brief
    *
    * This function takes in icellflag vector and sets velocity components inside the building
    * and terrain to zero.
    *
    */
    void setVelocityZero (URBGeneralData *UGD);


    /**
    * @brief
    *
    * This function takes in solver coefficient vectors (n, m, e, f, g and h)
    * and modify them for solver
    *
    */
    void solverCoefficients (URBGeneralData *UGD);

};
