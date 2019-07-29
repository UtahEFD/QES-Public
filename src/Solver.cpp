/*
 *
 * CUDA-URB
 * Copyright (c) 2019 Behnam Bozorgmehr
 * Copyright (c) 2019 Jeremy Gibbs
 * Copyright (c) 2019 Eric Pardyjak
 * Copyright (c) 2019 Zachary Patterson
 * Copyright (c) 2019 Rob Stoll
 * Copyright (c) 2019 Pete Willemsen
 *
 * This file is part of CUDA-URB package
 *
 * MIT License
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include "Solver.h"

using std::cerr;
using std::endl;
using std::vector;
using std::cout;

// duplication of this macro
#define CELL(i,j,k,w) ((i) + (j) * (nx+(w)) + (k) * (nx+(w)) * (ny+(w)))

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

/**< This function is showing progress of the solving process by printing the percentage */

void Solver::printProgress (float percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}


/**< \fn Solver
* This function is assigning values read by URBImputData to variables
* used in the solvers - this is only meant work with CUDA-URB!
 */

Solver::Solver(const URBInputData* UID, URBGeneralData * ugd)
    : eta( pow(alpha1/alpha2, 2.0) ),
      A( pow( ugd->dx/ugd->dy, 2.0 ) ),
      B( eta*pow( ugd->dx/ugd->dz, 2.0) ),
      itermax( UID->simParams->maxIterations )
{
    R.resize( ugd->numcell_cent, 0.0 );

    lambda.resize( ugd->numcell_cent, 0.0 );
    lambda_old.resize( ugd->numcell_cent, 0.0 );


/*
        // !!!!!!!!!!!!!!!!!!!!! @*@*@*@*@*@
        // We are here now...
        // where does this go then????

        // If there is wake behind the building to apply
        if (UID->simParams->wakeFlag > 0)
        {
          std::cout << "Applying wake behind building parameterization...\n";
          for (size_t pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
          {
            poly_buildings[pIdx].polygonWake (shpPolygons[pIdx], building_height[pIdx], base_height[pIdx], dx, dy, dz, z, nx, ny, nz,
                                          cavity_factor, wake_factor, dxy, icellflag, u0, v0, w0, max_velmag);

            //std::cout << "building added" << pIdx << std::endl;
          }
          std::cout << "Wake behind building parameterization done...\n";
        }

        auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
        std::chrono::duration<float> elapsedBuilding = finish - buildingsetup;
        std::cout << "Elapsed time to read in and create buildings and apply parameterization: " << elapsedBuilding.count() << " s\n";   // Print out elapsed execution time



    ///////////////////////////////////////////////////////////////
    //    Stair-step (original QUIC) for rectangular buildings   //
    ///////////////////////////////////////////////////////////////
    if (UID->simParams->meshTypeFlag == 0)
    {
        for (int i = 0; i < buildings.size(); i++)
        {
            ((RectangularBuilding*)buildings[i])->setCellsFlag(dx, dy, dz_array, nx, ny, nz, z, icellflag, UID->simParams->meshTypeFlag);
        }
    }
    /////////////////////////////////////////////////////////////
    //        Cut-cell method for rectangular buildings        //
    /////////////////////////////////////////////////////////////
    else
    {
      std::vector<std::vector<std::vector<float>>> x_cut(ugd->numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
      std::vector<std::vector<std::vector<float>>> y_cut(ugd->numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
      std::vector<std::vector<std::vector<float>>> z_cut(ugd->numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));

      std::vector<std::vector<int>> num_points(ugd->numcell_cent, std::vector<int>(6,0));
      std::vector<std::vector<float>> coeff(ugd->numcell_cent, std::vector<float>(6,0.0));

    	for (size_t i = 0; i < buildings.size(); i++)
    	{
        // Sets cells flag for each building
        ((RectangularBuilding*)buildings[i])->setCellsFlag(dx, dy, dz_array, nx, ny, nz, z, icellflag, UID->simParams->meshTypeFlag);
        ((RectangularBuilding*)buildings[i])->setCutCells(dx, dy, dz_array,z, nx, ny, nz, icellflag, x_cut, y_cut, z_cut,
                                                              num_points, coeff);    // Sets cut-cells for specified building,
                                                                                     // located in RectangularBuilding.h

      }

      if (buildings.size()>0)
      {
        /// Boundary condition for building edges
        calculateCoefficients(dx, dy, dz, nx, ny, nz, icellflag, n.data(), m.data(), f.data(), e.data(), h.data(), g.data(),
                                x_cut, y_cut, z_cut, num_points, coeff);
      }
    }*/



}

/*void Solver::calculateCoefficients(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag,
                        float* n, float* m, float* f, float* e, float* h, float* g,
                        std::vector<std::vector<std::vector<float>>> x_cut, std::vector<std::vector<std::vector<float>>>y_cut,
                        std::vector<std::vector<std::vector<float>>> z_cut, std::vector<std::vector<int>> num_points,
                        std::vector<std::vector<float>> coeff)

{

	for ( int k = 1; k < nz-2; k++)
	{
		for (int j = 1; j < ny-2; j++)
		{
			for (int i = 1; i < nx-2; i++)
			{
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);

				if (icellflag[icell_cent]==7)
				{
					for (int ii=0; ii<6; ii++)
					{
						coeff[icell_cent][ii] = 0;
						if (num_points[icell_cent][ii] !=0)
						{
							/// calculate area fraction coeeficient for each face of the cut-cell
							for (int jj=0; jj<num_points[icell_cent][ii]-1; jj++)
							{
								coeff[icell_cent][ii] += (0.5*(y_cut[icell_cent][ii][jj+1]+y_cut[icell_cent][ii][jj])*
														(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dy*dz) +
														(0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*
														(z_cut[icell_cent][ii][jj+1]-z_cut[icell_cent][ii][jj]))/(dx*dz) +
														(0.5*(x_cut[icell_cent][ii][jj+1]+x_cut[icell_cent][ii][jj])*
														(y_cut[icell_cent][ii][jj+1]-y_cut[icell_cent][ii][jj]))/(dx*dy);
							}

				coeff[icell_cent][ii] +=(0.5*(y_cut[icell_cent][ii][0]+y_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*
									(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dy*dz)+
									(0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*
									(z_cut[icell_cent][ii][0]-z_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dz)+
									(0.5*(x_cut[icell_cent][ii][0]+x_cut[icell_cent][ii][num_points[icell_cent][ii]-1])*
									(y_cut[icell_cent][ii][0]-y_cut[icell_cent][ii][num_points[icell_cent][ii]-1]))/(dx*dy);

						}
            coeff[icell_cent][ii] = 1;

					}

          /// Assign solver coefficients
					f[icell_cent] = coeff[icell_cent][0];
					e[icell_cent] = coeff[icell_cent][1];
					h[icell_cent] = coeff[icell_cent][2];
					g[icell_cent] = coeff[icell_cent][3];
					n[icell_cent] = coeff[icell_cent][4];
					m[icell_cent] = coeff[icell_cent][5];
				}

			}
		}
	}
}

*/
