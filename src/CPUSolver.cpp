#include "CPUSolver.h"

using std::cerr;
using std::endl;
using std::vector;
using std::cout;

void CPUSolver::solve(const URBInputData *UID, const URBGeneralData* ugd, bool solveWind)
{
    auto startOfSolveMethod = std::chrono::high_resolution_clock::now(); // Start recording execution time

    /////////////////////////////////////////////////////////////////
    ////////      Divergence of the initial velocity field   ////////
    /////////////////////////////////////////////////////////////////

    for (int k = 1; k < ugd->nz-1; k++)
    {
        for (int j = 0; j < ugd->ny-1; j++)
        {
            for (int i = 0; i < ugd->nx-1; i++)
            {
                icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);
                icell_face = i + j*ugd->nx + k*ugd->nx*ugd->ny;

                /// Calculate divergence of initial velocity field
                R[icell_cent] = (-2*pow(alpha1, 2.0))*((( e[icell_cent] * u0[icell_face+1]       - f[icell_cent] * u0[icell_face]) * dx ) +
                                                       (( g[icell_cent] * v0[icell_face + ugd->nx]    - h[icell_cent] * v0[icell_face]) * dy ) +
                                                       ( m[icell_cent]  * w0[icell_face + ugd->nx*ugd->ny] * dz_array[k]*0.5*(dz_array[k]+dz_array[k+1])
                                                        - n[icell_cent] * w0[icell_face] * dz_array[k]*0.5*(dz_array[k]+dz_array[k-1]) ));
            }
        }
    }

    if (solveWind)
    {
        auto startSolveSection = std::chrono::high_resolution_clock::now();

        //INSERT CANOPY CODE

        /////////////////////////////////////////////////
        //                 SOR solver              //////
        /////////////////////////////////////////////////
        int iter = 0;
        double error = 1.0;
    	  double reduced_error = 0.0;

        std::cout << "Solving...\n";
        while (iter < itermax && error > tol && error > reduced_error) {

            // Save previous iteration values for error calculation
            //    uses stl vector's assignment copy function, assign
            lambda_old.assign( lambda.begin(), lambda.end() );

            //
            // main SOR formulation loop
            //
            for (int k = 1; k < ugd->nz-2; k++){
            	for (int j = 1; j < ugd->ny-2; j++){
            	    for (int i = 1; i < ugd->nx-2; i++){

            	        icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);   /// Lineralized index for cell centered values

                        lambda[icell_cent] = (omega / ( e[icell_cent] + f[icell_cent] + g[icell_cent] +
                                                        h[icell_cent] + m[icell_cent] + n[icell_cent])) *
                            ( e[icell_cent] * lambda[icell_cent+1]        + f[icell_cent] * lambda[icell_cent-1] +
                              g[icell_cent] * lambda[icell_cent + (ugd->nx-1)] + h[icell_cent] * lambda[icell_cent-(ugd->nx-1)] +
                              m[icell_cent] * lambda[icell_cent+(ugd->nx-1)*(ugd->ny-1)] +
                              n[icell_cent] * lambda[icell_cent-(ugd->nx-1)*(ugd->ny-1)] - R[icell_cent] ) +
                            (1.0 - omega) * lambda[icell_cent];    /// SOR formulation
                    }
                }
            }

            /// Mirror boundary condition (lambda (@k=0) = lambda (@k=1))
            for (int j = 0; j < ugd->ny-1; j++){
                for (int i = 0; i < ugd->nx-1; i++){
                    int icell_cent = i + j*(ugd->nx-1);         /// Lineralized index for cell centered values
                    lambda[icell_cent] = lambda[icell_cent + (ugd->nx-1)*(ugd->ny-1)];
                }
            }

            /// Error calculation
            error = 0.0;                   /// Reset error value before error calculation
            for (int k = 0; k < ugd->nz-1; k++){
                for (int j = 0; j < ugd->ny-1; j++){
                    for (int i = 0; i < ugd->nx-1; i++){
                        int icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);   /// Lineralized index for cell centered values
                        error += fabs(lambda[icell_cent] - lambda_old[icell_cent])/((ugd->nx-1)*(ugd->ny-1)*(ugd->nz-1));
                    }
                }
            }

            if (iter == 0) {
                reduced_error = error * 1.0e-3;
            }

            iter += 1;
        }
        std::cout << "Solved!\n";

        std::cout << "Number of iterations:" << iter << "\n";   // Print the number of iterations
        std::cout << "Error:" << error << "\n";
        std::cout << "Reduced Error:" << reduced_error << "\n";

        ////////////////////////////////////////////////////////////////////////
        /////   Update the velocity field using Euler-Lagrange equations   /////
        ////////////////////////////////////////////////////////////////////////

        for (int k = 0; k < ugd->nz; k++)
        {
            for (int j = 0; j < ugd->ny; j++)
            {
                for (int i = 0; i < ugd->nx; i++)
                {
                    int icell_face = i + j*ugd->nx + k*ugd->nx*ugd->ny;   /// Lineralized index for cell faced values
                    u[icell_face] = u0[icell_face];
                    v[icell_face] = v0[icell_face];
                    w[icell_face] = w0[icell_face];
                }
            }
        }


        // /////////////////////////////////////////////
    	  /// Update velocity field using Euler equations
        // /////////////////////////////////////////////
        for (int k = 1; k < ugd->nz-1; k++)
        {
            for (int j = 1; j < ugd->ny-1; j++)
            {
                for (int i = 1; i < ugd->nx-1; i++)
                {
                    icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);   /// Lineralized index for cell centered values
                    icell_face = i + j*ugd->nx + k*ugd->nx*ugd->ny;               /// Lineralized index for cell faced values

                    // Calculate correct wind velocity
                    u[icell_face] = u0[icell_face] + (1/(2*pow(alpha1, 2.0))) *
                        f[icell_cent]*dx*(lambda[icell_cent]-lambda[icell_cent-1]);

                    v[icell_face] = v0[icell_face] + (1/(2*pow(alpha1, 2.0))) *
                        h[icell_cent]*dy*(lambda[icell_cent]-lambda[icell_cent - (ugd->nx-1)]);

                    w[icell_face] = w0[icell_face]+(1/(2*pow(alpha2, 2.0))) *
                        n[icell_cent]*dz_array[k]*(lambda[icell_cent]-lambda[icell_cent - (ugd->nx-1)*(ugd->ny-1)]);
                }
            }
        }

        for (int k = 1; k < ugd->nz-1; k++)
        {
            for (int j = 0; j < ugd->ny-1; j++)
            {
                for (int i = 0; i < ugd->nx-1; i++)
                {
                    icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);   /// Lineralized index for cell centered values
                    icell_face = i + j*ugd->nx + k*ugd->nx*ugd->ny;               /// Lineralized index for cell faced values

                    // If we are inside a building, set velocities to 0.0
                    if (icellflag[icell_cent] == 0 || icellflag[icell_cent] == 2)
                    {
                        /// Setting velocity field inside the building to zero
                        u[icell_face] = 0;
                        u[icell_face+1] = 0;
                        v[icell_face] = 0;
                        v[icell_face+ugd->nx] = 0;
                        w[icell_face] = 0;
                        w[icell_face+ugd->nx*ugd->ny] = 0;
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
