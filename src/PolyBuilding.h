#pragma	once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <algorithm>

#include "util/ParseInterface.h"

#include "Building.h"


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
    std::vector<float> xf1, yf1, xf2, yf2;
    int icell_cent, icell_face;
    float x1, x2, y1, y2;
    std::vector<float> upwind_rel_dir;


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
    * This constructor creates a polygon type building: calculates and initializes
    * all the features specific to this type of the building. This function reads in
    * nodes of the polygon along with height and base height of the building.
    *
    */
    PolyBuilding(const URBInputData* UID, URBGeneralData* UGD, int id);


    // Need to complete!!!
    virtual void parseValues() {}


    /**
     *
     */
    void setPolyBuilding(URBGeneralData* UGD);


    /**
    *
    * This function defines bounds of the polygon building and sets the icellflag values
    * for building cells. It applies the Stair-step method to define building bounds.
    *
    */
    void setCellFlags(const URBInputData* UID, URBGeneralData* UGD, int building_number);

    /**
    *
    * This function applies the upwind cavity in front of the building to buildings defined as polygons.
    * This function reads in building features like nodes, building height and base height and uses
    * features of the building defined in the class constructor and setCellsFlag function. It defines
    * cells in the upwind area and applies the approperiate parameterization to them.
    * More information: "Improvements to a fast-response urban wind model, M. Nelson et al. (2008)"
    *
    */
    void upwindCavity (const URBInputData* UID, URBGeneralData* UGD);

    /**
    *
    * This function applies wake behind the building parameterization to buildings defined as polygons.
    * The parameterization has two parts: near wake and far wake. This function reads in building features
    * like nodes, building height and base height and uses features of the building defined in the class
    * constructor ans setCellsFlag function. It defines cells in each wake area and applies the approperiate
    * parameterization to them.
    *
    */
    void polygonWake (const URBInputData* UID, URBGeneralData* UGD, int building_id);


    /**
    *
    * This function applies the street canyon parameterization to the qualified space between buildings defined as polygons.
    * This function reads in building features like nodes, building height and base height and uses
    * features of the building defined in the class constructor and setCellsFlag function. It defines
    * cells qualified in the space between buildings and applies the approperiate parameterization to them.
    * More information: "Improvements to a fast-response urban wind model, M. Nelson et al. (2008)"
    *
    */
    void streetCanyon (URBGeneralData *UGD);


    /**
    *
    * This function applies the sidewall parameterization to the qualified space on the side of buildings defined as polygons.
    * This function reads in building features like nodes, building height and base height and uses
    * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
    * cells qualified on the side of buildings and applies the approperiate parameterization to them.
    * More information: "Comprehensive Evaluation of Fast-Response, Reynolds-Averaged Navier–Stokes, and Large-Eddy Simulation
    * Methods Against High-Spatial-Resolution Wind-Tunnel Data in Step-Down Street Canyons, A. N. Hayati et al. (2017)"
    *
    */
    void sideWall (const URBInputData* UID, URBGeneralData* UGD);


    /**
    *
    * This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
    * This function reads in building features like nodes, building height and base height and uses
    * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
    * cells qualified on top of buildings and applies the approperiate parameterization to them.
    * More information:
    *
    */
    void rooftop (const URBInputData* UID, URBGeneralData* UGD);


    /**
    *
    * This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
    * This function reads in building features like nodes, building height and base height and uses
    * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
    * cells qualified on top of buildings and applies the approperiate parameterization to them.
    * More information:
    *
    */
    //void streetIntersection (const URBInputData* UID, URBGeneralData* UGD);


    /**
    *
    * This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
    * This function reads in building features like nodes, building height and base height and uses
    * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
    * cells qualified on top of buildings and applies the approperiate parameterization to them.
    * More information:
    *
    */
    //void poisson (const URBInputData* UID, URBGeneralData* UGD);

    void reorderPoints(std::vector <cutVert> &face_points, int index);

    void mergeSort(std::vector<float> &angle, std::vector <cutVert> &face_points);


    float calculateArea( URBGeneralData* UGD, std::vector <cutVert> &face_points, int cutcell_index, int index);


    /**
    *
    * This function applies the non local mixing length model.
    * More information: William et al. 2004
    *
    */
    void NonLocalMixing (URBGeneralData* UGD, TURBGeneralData* TGD);

};
