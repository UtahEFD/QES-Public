#pragma once

// may need to forward reference this???
class URBGeneralData;

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

};
