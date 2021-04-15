/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file CPUSolver.cpp */

#include "CPUSolver.h"

using std::cerr;
using std::endl;
using std::vector;
using std::cout;


/** :document this:
 * Start by writing a one sentence description here
 *
 * Complete by continuting to write implementation details here.
 * (remove :document this: tag when done)
 */
void CPUSolver::solve(const WINDSInputData* WID, WINDSGeneralData* WGD, bool solveWind)
{
    auto startOfSolveMethod = std::chrono::high_resolution_clock::now(); // Start recording execution time

    /***************************************************************
     *********   Divergence of the initial velocity field   ********
     ***************************************************************/

    int icell_face;   // cell-face index
    int icell_cent;   // cell-centered index

    R.resize( WGD->numcell_cent, 0.0 );

    lambda.resize( WGD->numcell_cent, 0.0 );
    lambda_old.resize( WGD->numcell_cent, 0.0 );

    for (int k = 1; k < WGD->nz-2; k++)
    {
        for (int j = 0; j < WGD->ny-1; j++)
        {
            for (int i = 0; i < WGD->nx-1; i++)
            {
                icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;

                // Calculate divergence of initial velocity field
                R[icell_cent] = (-2*pow(alpha1, 2.0))*((( WGD->u0[icell_face+1]       - WGD->u0[icell_face]) / WGD->dx ) +
                                                       (( WGD->v0[icell_face + WGD->nx]    - WGD->v0[icell_face]) / WGD->dy ) +
                                                       (( WGD->w0[icell_face + WGD->nx*WGD->ny] - WGD->w0[icell_face]) / WGD->dz_array[k] ));
            }
        }
    }




    if (solveWind)
    {
        auto startSolveSection = std::chrono::high_resolution_clock::now();

        // INSERT CANOPY CODE

        /***************************************************************
         **********************   SOR Solver   *************************
         ***************************************************************/

        int iter = 0;
        float error;
        float max_error = 1.0;

        std::cout << "Solving...\n";
        while (iter < itermax && max_error > tol) {

            // Save previous iteration values for error calculation
            //    uses stl vector's assignment copy function, assign
            lambda_old.assign( lambda.begin(), lambda.end() );

            //
            // main SOR formulation loop
            //
            for (int k = 1; k < WGD->nz-2; k++){
            	for (int j = 1; j < WGD->ny-2; j++){
            	    for (int i = 1; i < WGD->nx-2; i++){

            	        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);   // Lineralized index for cell centered values

                        lambda[icell_cent] = (omega / ( WGD->e[icell_cent] + WGD->f[icell_cent] + WGD->g[icell_cent] +
                                                        WGD->h[icell_cent] + WGD->m[icell_cent] + WGD->n[icell_cent])) *
                            ( WGD->e[icell_cent] * lambda[icell_cent+1]        + WGD->f[icell_cent] * lambda[icell_cent-1] +
                              WGD->g[icell_cent] * lambda[icell_cent + (WGD->nx-1)] + WGD->h[icell_cent] * lambda[icell_cent-(WGD->nx-1)] +
                              WGD->m[icell_cent] * lambda[icell_cent+(WGD->nx-1)*(WGD->ny-1)] +
                              WGD->n[icell_cent] * lambda[icell_cent-(WGD->nx-1)*(WGD->ny-1)] - R[icell_cent] ) +
                            (1.0 - omega) * lambda[icell_cent];    // SOR formulation

                    }
                }
            }

            // Mirror boundary condition (lambda (@k=0) = lambda (@k=1))
            for (int j = 0; j < WGD->ny-1; j++){
                for (int i = 0; i < WGD->nx-1; i++){
                    int icell_cent = i + j*(WGD->nx-1);         // Lineralized index for cell centered values
                    lambda[icell_cent] = lambda[icell_cent + (WGD->nx-1)*(WGD->ny-1)];
                }
            }

            // Error calculation
            max_error = 0.0;                   // Reset error value before error calculation
            for (int k = 0; k < WGD->nz-1; k++){
                for (int j = 0; j < WGD->ny-1; j++){
                    for (int i = 0; i < WGD->nx-1; i++){
                        int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);   // Lineralized index for cell centered values
                        error = fabs(lambda[icell_cent] - lambda_old[icell_cent]);
                        if (error > max_error)
                        {
                          max_error = error;
                        }
                    }
                }
            }

            iter += 1;
        }
        std::cout << "Solved!\n";

        std::cout << "Number of iterations:" << iter << "\n";   // Print the number of iterations
        std::cout << "Error:" << max_error << "\n";
        std::cout << "tol:" << tol << "\n";


        /***************************************************************
         *** Update the velocity field using Euler-Lagrange equations **
         ***************************************************************/

        for (int k = 0; k < WGD->nz-1; k++)
        {
            for (int j = 0; j < WGD->ny; j++)
            {
                for (int i = 0; i < WGD->nx; i++)
                {
                    int icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;   // Lineralized index for cell faced values
                    WGD->u[icell_face] = WGD->u0[icell_face];
                    WGD->v[icell_face] = WGD->v0[icell_face];
                    WGD->w[icell_face] = WGD->w0[icell_face];
                }
            }
        }


        /***************************************************************
         ******* Update the velocity field using Euler equations *******
         ***************************************************************/
        for (int k = 1; k < WGD->nz-2; k++)
        {
            for (int j = 1; j < WGD->ny-1; j++)
            {
                for (int i = 1; i < WGD->nx-1; i++)
                {
                    icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);   // Lineralized index for cell centered values
                    icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;               // Lineralized index for cell faced values

                    WGD->u[icell_face] = WGD->u0[icell_face] + (1/(2*pow(alpha1, 2.0))) *
                        WGD->f[icell_cent]*WGD->dx*(lambda[icell_cent]-lambda[icell_cent-1]);

                        // Calculate correct wind velocity
                    WGD->v[icell_face] = WGD->v0[icell_face] + (1/(2*pow(alpha1, 2.0))) *
                        WGD->h[icell_cent]*WGD->dy*(lambda[icell_cent]-lambda[icell_cent - (WGD->nx-1)]);

                    WGD->w[icell_face] = WGD->w0[icell_face]+(1/(2*pow(alpha2, 2.0))) *
                        WGD->n[icell_cent]*WGD->dz_array[k]*(lambda[icell_cent]-lambda[icell_cent - (WGD->nx-1)*(WGD->ny-1)]);
                }
            }
        }

        for (int k = 1; k < WGD->nz-1; k++)
        {
            for (int j = 0; j < WGD->ny-1; j++)
            {
                for (int i = 0; i < WGD->nx-1; i++)
                {
                    icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);   // Lineralized index for cell centered values
                    icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;               // Lineralized index for cell faced values

                    // If we are inside a building, set velocities to 0.0
                    if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2)
                    {
                        // Setting velocity field inside the building to zero
                        WGD->u[icell_face] = 0;
                        WGD->u[icell_face+1] = 0;
                        WGD->v[icell_face] = 0;
                        WGD->v[icell_face+WGD->nx] = 0;
                        WGD->w[icell_face] = 0;
                        WGD->w[icell_face+WGD->nx*WGD->ny] = 0;
                    }
                }
            }
        }

        auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
        std::chrono::duration<float> elapsedTotal = finish - startOfSolveMethod;
        std::chrono::duration<float> elapsedSolve = finish - startSolveSection;
        std::cout << "Elapsed total time: " << elapsedTotal.count() << " s\n";   // Print out elapsed execution time
        std::cout << "Elapsed solve time: " << elapsedSolve.count() << " s\n";   // Print out elapsed execution time

    }


}
