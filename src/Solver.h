#pragma once

/* 
 * This is an abstract class that is the basis for the windfield
 * convergence algorithm. This class has information needed to run
 * the simulation as well as functions widely used by different solver
 * methods
 */

#include "URBInputData.h"
#include "SimulationParameters.h"
#include "Buildings.h"
#include "NonPolyBuilding.h"
#include "RectangularBuilding.h"
#include "Vector3.h"
#include "NetCDFData.h"
#include "Mesh.h"
#include "DTEHeightField.h"
#include <math.h>
#include <vector>

#define MIN_S(x,y) ((x) < (y) ? (x) : (y))
#define MAX_S(x,y) ((x) > (y) ? (x) : (y))

class Solver
{
protected:
	int nx, ny, nz;
	float dx, dy, dz;
    float dxy;
    std::vector<float> dzArray, zm;
    std::vector<float> x,y,z;
	int itermax;

    float z_ref;             /// Height of the measuring sensor (m)
    float U_ref;             /// Measured velocity at the sensor height (m/s)
    float z0;
    float max_velmag;
    std::vector<Building*> buildings;

    Mesh* mesh;

    int rooftopFlag;
    int upwindCavityFlag;
    int streetCanyonFlag;
    int streetIntersectionFlag;
    int wakeFlag;
    int sidewallFlag;


    const int alpha1 = 1;        /// Gaussian precision moduli
    const int alpha2 = 1;        /// Gaussian precision moduli
    const float eta = pow(alpha1/alpha2, 2.0);
    const float A = pow(dx/dy, 2.0);
    const float B = eta*pow(dx/dz, 2.0);
    const float tol = 1e-9;     /// Error tolerance
    const float omega = 1.78;   /// Over-relaxation factor
    const float pi = 4.0f * atan(1.0);

    /*
     * This prints out the current amount that a process
     * has finished with a progress bar
     *
     * @param percentage -the amount the task has finished
     */
    void printProgress (float percentage);

public:

	Solver(URBInputData* UID, DTEHeightField* DTEHF);

    /* 
     * This is the function that sets up and runs the convergence algorithm
     * It is purely virtual and has no base functionality.
     *
     * @param netcdfDat -the netcdf file to send the final results to
     * @param solveWind -if the solver should be run or not
     * @param cellFace -if cellFace is being used or if cellCenter is being used 
     */
	virtual void solve(NetCDFData* netcdfDat, bool solveWind, bool cellFace) = 0;

    /*
     * Defines what the walls are concerning buildings and sets the approprirate values
     * in domain representations
     *
     * @param iCellFlag -the identifier for what a cell is at each cell in the domain
     * @param n -n values in the domain
     * @param m -m values in the domain
     * @param f -f values in the domain
     * @param e -e values in the domain
     * @param h -h values in the domain
     * @param g -g values in the domain
     */
    void defineWalls(int* iCellFlag, float* n, float* m, float* f, float* e, float* h, float* g);
    
    //not really done yet
    void upWind(Building* build, int* iCellFlag, double* u0, double* v0, double* w0, float* z, float* zm);
    //not really done yet
    void reliefWake(NonPolyBuilding* build, float* u0, float* v0);
};