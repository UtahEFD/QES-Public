#pragma	once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>

#include "util/ParseInterface.h"

#include "Building.h"
#include "PolygonVertex.h"

using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

#define MIN_S(x,y) ((x) < (y) ? (x) : (y))

/**
*
* This class is designed for the general building shape (polygons).
* It's an inheritance of the building class (has all the features defined in that class).
* In this class, first, the polygone buildings will be defined and then different
* parameterizations related to each building will be applied. For now, it only includes
* wake behind the building parameterization.
*
*/

class PolyBuilding : public Building
{
private:

protected:

    float x_min, x_max, y_min, y_max;      // Minimum and maximum values of x
                                           // and y for a building
    float x_cent, y_cent;                  /**< Coordinates of center of a cell */
    float polygon_area;                    /**< Polygon area */
    std::vector<float> xi, yi;
    int icell_cent, icell_face;
    float x1, x2, y1, y2;
    std::vector <polyVert> polygonVertices;

public:

    PolyBuilding()
        : Building()
    {
        // What should go here ???  Good to consider this case so we
        // don't have problems down the line.
    }

    virtual ~PolyBuilding() 
    {
    }
    
    /**
    *
    * Converts data about rect building into something that poly can
    * building can use
    *
    * This constructor creates a polygon type building using
    * rectangular building information...
    *
    */
    PolyBuilding( const URBInputData* UID, URBGeneralData* UGD, float x_start,
                  float y_start, float base_height, float L, float W, float H,
                  float building_rotation);


    PolyBuilding( float x_start, float y_start, float base_height, float L, float W,
                  float H, float canopy_rotation, const URBInputData* UID, URBGeneralData* UGD);


    /**
    *
    * This constructor creates a polygon type building: calculates and initializes
    * all the features specific to this type of the building. This function reads in
    * nodes of the polygon along with height and base height of the building.
    *
    */
    PolyBuilding(const URBInputData* UID, URBGeneralData* UGD, int id);


    // Need to complete!!!
    virtual void parseValues() {}


    // make sure virtual functions from "Building" get implemented
    // here ...


    /**
     *
     */
    void setPolybuilding(int nx, int ny, float dx, float dy, std::vector<double> &u0, std::vector<double> &v0, std::vector<float> z);


    /**
    *
    * This function defines bounds of the polygon building and sets the icellflag values
    * for building cells. It applies the Stair-step method to define building bounds.
    *
    */
    void setCellFlags(const URBInputData* UID, URBGeneralData* UGD);

    /**
    *
    * This function applies wake behind the building parameterization to buildings defined as polygons.
    * The parameterization has two parts: near wake and far wake. This function reads in building features
    * like nodes, building height and base height and uses features of the building defined in the class
    * constructor ans setCellsFlag function. It defines cells in each wake area and applies the approperiate
    * parameterization to them.
    *
    */
    void polygonWake (const URBInputData* UID, URBGeneralData* UGD);

};
