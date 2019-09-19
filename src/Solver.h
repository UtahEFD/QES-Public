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



#include "URBInputData.h"
#include "URBGeneralData.h"

#include "Vector3.h"

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
    // Matthieu will use change them in future
    const int alpha1;        /**< Gaussian precision moduli */
    const int alpha2;        /**< Gaussian precision moduli */
    const float eta; //= pow((alpha1/alpha2), 2.0);
    const float A; //= pow(UGD->dx/UGD-dy, 2.0);
    const float B; //= eta*pow(dx/dz, 2.0);

    const float tol = 1.0e-9;     /**< Error tolerance -->>> very
                                   * small tol for float! */
    const float omega = 1.78f;   /**< Over-relaxation factor */

    // SOLVER-based parameters
    std::vector<double> R;           /**< Divergence of initial velocity field */
    std::vector<double> lambda, lambda_old;



    int itermax;		/**< Maximum number of iterations */


    /// Declaration of Lagrange multipliers



    ////////////////////////////////////////////////////////////////////////////
    //////// Variables and constants needed in solver and other functions-- Behnam
    ////////////////////////////////////////////////////////////////////////////


    /*
     * This prints out the current amount that a process
     * has finished with a progress bar
     *
     * @param percentage -the amount the task has finished
     */
    void printProgress (float percentage);


public:
    Solver(const URBInputData* UID, URBGeneralData* UGD);

    virtual void solve(const URBInputData *UID, URBGeneralData* UGD, bool solveWind) = 0;

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

};
