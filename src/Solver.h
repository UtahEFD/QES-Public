#pragma once

/*
 * This is an abstract class that is the basis for the windfield
 * convergence algorithm. This class has information needed to run
 * the simulation as well as functions widely used by different solver
 * methods
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>
#include <limits>

#include "Cut_cell.h"
#include "PolyBuilding.h"

#include "URBInputData.h"
#include "URBGeneralData.h"

#include "SimulationParameters.h"
#include "Buildings.h"
#include "NonPolyBuilding.h"
#include "RectangularBuilding.h"
#include "Vector3.h"
#include "Output.hpp"

#include "Cell.h"
#include "Sensor.h"
#include "Canopies.h"
#include "Canopy.h"

using namespace std;

/**< \class Solver
* This class declares and defines variables required for both solvers
*
* There are several special member variables that should be accessible
* to all solvers.  They are declared in this class.
*/
class Solver
{
protected:

    ////////////////////////////////////////////////////////////////////////////
    //////// Variables and constants needed only in solver -- Behnam
    //////// These can be stayed in the solver class
    ////////////////////////////////////////////////////////////////////////////
    const float eta = pow(alpha1/alpha2, 2.0);
    const float A = pow(dx/dy, 2.0);
    const float B = eta*pow(dx/dz, 2.0);
    const float tol = 1.0e-9;     /**< Error tolerance -->>> very
                                   * small tol for float! */
    const float omega = 1.78f;   /**< Over-relaxation factor */

    // SOLVER-based parameters
    std::vector<double> R;           /**< Divergence of initial velocity field */
    std::vector<double> lambda, lambda_old;    

    /// Declaration of final velocity field components (u,v,w)
    std::vector<double> u,v,w;
    
    int itermax;		/**< Maximum number of iterations */


    /// Declaration of Lagrange multipliers



    ////////////////////////////////////////////////////////////////////////////
    //////// Variables and constants needed in solver and other functions-- Behnam
    ////////////////////////////////////////////////////////////////////////////
    // Matthieu will use change them in future
    const int alpha1 = 1;        /**< Gaussian precision moduli */
    const int alpha2 = 1;        /**< Gaussian precision moduli */


    



    /*
     * This prints out the current amount that a process
     * has finished with a progress bar
     *
     * @param percentage -the amount the task has finished
     */
    void printProgress (float percentage);


public:
    Solver(const URBInputData* UID, const DTEHeightField* DTEHF, URBGeneralData* ugd, Output* output);

    virtual void solve(bool solveWind) = 0;

    /**
     * @brief
     *
     * This function takes in values necessary for cut-cell method for
     * buildings and then calculates the area fraction coefficients,
     * sets them to approperiate solver coefficients and finally sets
     * related coefficients to zero to define solid walls for non
     * cut-cells.
     */
    void calculateCoefficients(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag, float* n, float* m,
                     float* f, float* e, float* h, float* g, std::vector<std::vector<std::vector<float>>> x_cut,
                     std::vector<std::vector<std::vector<float>>>y_cut,std::vector<std::vector<std::vector<float>>> z_cut,
                     std::vector<std::vector<int>> num_points, std::vector<std::vector<float>> coeff);
    /**
     * @brief
     *
     * This function takes in the icellflags set by setCellsFlag
     * function for stair-step method and sets related coefficients to
     * zero to define solid walls.
     */
    void defineWalls(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag, float* n, float* m,
                     float* f, float* e, float* h, float* g);


    /**
    * @brief
    *
    * This function takes in the icellflag valus set by setCellsFlag
    * function for both stair-step and cut-cell methods and creates vectors of indices
    * of the cells that have wall to right/left, wall above/bellow and wall in front/back
    *
    */
    void getWallIndices (std::vector<int>& icellflag, std::vector<int>& wall_right_indices, std::vector<int>& wall_left_indices,
                        std::vector<int>& wall_above_indices, std::vector<int>& wall_below_indices,
                        std::vector<int>& wall_front_indices, std::vector<int>& wall_back_indices);

    /**
    * @brief
    *
    * This function takes in vectores of indices of the cells that have wall to right/left,
    * wall above/bellow and wall in front/back and applies the log law boundary condition fix
    * to the cells near Walls (based on Kevin Briggs master's thesis (2015))
    *
    */
    void wallLogBC (std::vector<int>& wall_right_indices, std::vector<int>& wall_left_indices,
                    std::vector<int>& wall_above_indices, std::vector<int>& wall_below_indices,
                    std::vector<int>& wall_front_indices, std::vector<int>& wall_back_indices,
                    double *u0, double *v0, double *w0, float z0);


    void mergeSort( std::vector<float> &height, std::vector<std::vector<polyVert>> &poly_points, std::vector<float> &base_height, std::vector<float> &building_height);

    /**
    * @brief
    *
    * This function saves user-defined data to file
    */
    void save(Output*);
};
