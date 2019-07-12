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

#include "ESRIShapefile.h"

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

Solver::Solver(const URBInputData* UID,
               const DTEHeightField* DTEHF,
               UrbGeneralData * ugd,
               Output* output)
    : itermax( UID->simParams->maxIterations )
{
    R.resize( ugd->numcell_cent, 0.0 );
    
    lambda.resize( ugd->numcell_cent, 0.0 );
    lambda_old.resize( numcell_cent, 0.0 );
    
    u.resize(numcell_face, 0.0);
    v.resize(numcell_face, 0.0);
    w.resize(numcell_face, 0.0);

    ////////////////////////////////////////////////////////
    //
    // THIS NEEDS TO BE MOVED OUT OF HERE AND INTO THE INPUT PARSING!!!
    // -Pete
    // 
    // Behnam also notes that this section will be completely changed
    // to NOT treat terrain cells as "buildings" -- Behnam will fix
    // this
    // 
    ////////////////////////////////////////////////////////
    //////              Apply Terrain code             /////
    ///////////////////////////////////////////////////////
    mesh = 0;
    if (DTEHF)
    {

        DTEHFExists = true;
        mesh = new Mesh(DTEHF->getTris());

        // ////////////////////////////////
        // Retrieve terrain height field //
        // ////////////////////////////////
        for (int i = 0; i < nx-1; i++)
        {
            for (int j = 0; j < ny-1; j++)
            {
              // Gets height of the terrain for each cell
              int idx = i + j*(nx-1);
              terrain[idx] = mesh->getHeight(i * dx + dx * 0.5f, j * dy + dy * 0.5f);
              if (terrain[idx] < 0.0)
              {
                terrain[idx] = 0.0;
              }
              id = i+j*nx;
              for (auto k=0; k<z.size(); k++)
        			{
        				terrain_id[id] = k+1;
        				if (terrain[idx] < z[k+1])
        				{
        					break;
        				}
        			}
            }
        }

        if (UID->simParams->meshTypeFlag == 0)
        {
            // ////////////////////////////////
            // Stair-step (original QUIC)    //
            // ////////////////////////////////
            if (mesh)
            {
                std::cout << "Creating terrain blocks...\n";
                for (int i = 0; i < nx-1; i++)
                {
                    for (int j = 0; j < ny-1; j++)
                    {
                      // Gets height of the terrain for each cell
                      float heightToMesh = mesh->getHeight(i * dx + dx * 0.5f, j * dy + dy * 0.5f);
                      // Calls rectangular building to create a each cell to the height of terrain in that cell
                      if (heightToMesh > z[1])
                      {
                        buildings.push_back(new RectangularBuilding(i * dx+UID->simParams->halo_x, j * dy+UID->simParams->halo_y, 0.0, dx, dy, heightToMesh,z));
                      }

                    }
                    printProgress( (float)i / (float)nx);
                }
                std::cout << "blocks created\n";
            }
        }
        else
        {
            // ////////////////////////////////
            //        Cut-cell method        //
            // ////////////////////////////////

            // Calling calculateCoefficient function to calculate area fraction coefficients for cut-cells
            cut_cell.calculateCoefficient(cells, DTEHF, nx, ny, nz, dx, dy, dz, n, m, f, e, h, g, pi, icellflag);
        }
    }
    ///////////////////////////////////////////////////////
    //////   END END END of  Apply Terrain code       /////
    ///////////////////////////////////////////////////////


    // now here...

    /// defining ground solid cells (ghost cells below the surface)
    for (int j = 0; j < ny-1; j++)
    {
        for (int i = 0; i < nx-1; i++)
        {
            int icell_cent = i + j*(nx-1);
            icellflag[icell_cent] = 0.0;
        }
    }

    /////////////////////////////////////////////////////////////
    //                Apply building effect                    //
    /////////////////////////////////////////////////////////////


    // For now, process ESRIShapeFile here:
    ESRIShapefile *shpFile = nullptr;

    if (UID->simParams->shpFile != "")
    {
      auto buildingsetup = std::chrono::high_resolution_clock::now(); // Start recording execution time

        std::vector <PolyBuilding> poly_buildings;
        std::vector< std::vector <polyVert> > shpPolygons;
        std::vector< std::vector <polyVert> > poly;
        std::vector <float> base_height;            // Base height of buildings
        std::vector <float> effective_height;            // Effective height of buildings
        float corner_height, min_height;
        std::vector <float> building_height;        // Height of buildings

        // Read polygon node coordinates and building height from shapefile
        shpFile = new ESRIShapefile( UID->simParams->shpFile,
                                     UID->simParams->shpBuildingLayerName,
                                     shpPolygons, building_height );



        std::vector<float> shpDomainSize(2), minExtent(2);
        shpFile->getLocalDomain( shpDomainSize );
        shpFile->getMinExtent( minExtent );

        float domainOffset[2] = { 0, 0 };
        for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
        {
            // convert the global polys to local domain coordinates
          for (auto lIdx=0; lIdx<shpPolygons[pIdx].size(); lIdx++)
          {
            shpPolygons[pIdx][lIdx].x_poly = shpPolygons[pIdx][lIdx].x_poly - minExtent[0] ;
            shpPolygons[pIdx][lIdx].y_poly = shpPolygons[pIdx][lIdx].y_poly - minExtent[1] ;
          }
        }
        std::cout << "num_poly buildings" << shpPolygons.size() << std::endl;
        // Setting base height for buildings if there is a DEM file
        if (UID->simParams->demFile != "")
        {
          for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
          {
            // Get base height of every corner of building from terrain height
            min_height = mesh->getHeight(shpPolygons[pIdx][0].x_poly, shpPolygons[pIdx][0].y_poly);
            if (min_height<0)
            {
              min_height = 0.0;
            }
            for (auto lIdx=1; lIdx<shpPolygons[pIdx].size(); lIdx++)
            {
              corner_height = mesh->getHeight(shpPolygons[pIdx][lIdx].x_poly, shpPolygons[pIdx][lIdx].y_poly);
              if (corner_height<min_height && corner_height>0.0)
              {
                min_height = corner_height;
              }
            }
            base_height.push_back(min_height);
          }
        }
        else
        {
          for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
          {
            base_height.push_back(0.0);
          }
        }

        for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
        {
          effective_height.push_back (base_height[pIdx]+building_height[pIdx]);
          for (auto lIdx=0; lIdx<shpPolygons[pIdx].size(); lIdx++)
          {
            shpPolygons[pIdx][lIdx].x_poly += UID->simParams->halo_x;
            shpPolygons[pIdx][lIdx].y_poly += UID->simParams->halo_y;
          }
        }

        mergeSort( effective_height, shpPolygons, base_height, building_height);
        std::cout << "Creating buildings from shapefile...\n";
        // Loop to create each of the polygon buildings read in from the shapefile
        for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
        {
          // Create polygon buildings
          poly_buildings.push_back (PolyBuilding (shpPolygons[pIdx], building_height[pIdx], base_height[pIdx], nx, ny,
                                      nz, dx, dy, dz, u0, v0, z));
        }

        for (auto pIdx = 0; pIdx<shpPolygons.size(); pIdx++)
        {
          // Call setCellsFlag in the PolyBuilding class to identify building cells
          poly_buildings[pIdx].setCellsFlag ( dx, dy, dz, z, nx, ny, nz, icellflag, UID->simParams->meshTypeFlag, shpPolygons[pIdx], base_height[pIdx], building_height[pIdx]);
        }
        std::cout << "Buildings created...\n";

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
    }



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
      std::vector<std::vector<std::vector<float>>> x_cut(numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
      std::vector<std::vector<std::vector<float>>> y_cut(numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));
      std::vector<std::vector<std::vector<float>>> z_cut(numcell_cent, std::vector<std::vector<float>>(6, std::vector<float>(6,0.0)));

      std::vector<std::vector<int>> num_points(numcell_cent, std::vector<int>(6,0));
      std::vector<std::vector<float>> coeff(numcell_cent, std::vector<float>(6,0.0));

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
    }

    std::cout << "Defining Solid Walls...\n";
    /// Boundary condition for building edges
    defineWalls(dx,dy,dz,nx,ny,nz, icellflag, n.data(), m.data(), f.data(), e.data(), h.data(), g.data());
    std::cout << "Walls Defined...\n";


    /*
     * Calling getWallIndices to return 6 vectores of indices of the cells that have wall to right/left,
     * wall above/below and wall in front/back
     */
    getWallIndices (icellflag, wall_right_indices, wall_left_indices, wall_above_indices,
      wall_below_indices, wall_front_indices, wall_back_indices);

    /*
     * Calling wallLogBC to read in vectores of indices of the cells that have wall to right/left,
     * wall above/below and wall in front/back and applies the log law boundary condition fix
     * to the cells near Walls
     *
     */
    /*wallLogBC (wall_right_indices, wall_left_indices, wall_above_indices, wall_below_indices,
      wall_front_indices, wall_back_indices, u0.data(), v0.data(), w0.data(), z0);*/

      for (int k = 1; k < nz-1; k++)
      {
          for (int j = 1; j < ny-1; j++)
          {
              for (int i = 1; i < nx-1; i++)
              {
                  icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                  icell_face = i + j*nx + k*nx*ny;
                  if (icellflag[icell_cent] == 0) {
                      u0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
                      u0[icell_face+1] = 0.0;
                      v0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
                      v0[icell_face+nx] = 0.0;
                      w0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
                      w0[icell_face+nx*ny] = 0.0;
                  }
              }
          }
      }

    /// New boundary condition implementation
    for (int k = 0; k < nz-1; k++)
    {
        for (int j = 0; j < ny-1; j++)
        {
            for (int i = 0; i < nx-1; i++)
            {
                icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                e[icell_cent] = e[icell_cent]/(dx*dx);
                f[icell_cent] = f[icell_cent]/(dx*dx);
                g[icell_cent] = g[icell_cent]/(dy*dy);
                h[icell_cent] = h[icell_cent]/(dy*dy);
                m[icell_cent] = m[icell_cent]/(dz_array[k]*0.5*(dz_array[k]+dz_array[k+1]));
                n[icell_cent] = n[icell_cent]/(dz_array[k]*0.5*(dz_array[k]+dz_array[k-1]));
            }
        }
    }

    //////////////////////////////////////////////////
    //      Initialize output information           //
    //////////////////////////////////////////////////

    if (output != nullptr) {

        // set output fields
        std::cout<<"Getting output fields"<<std::endl;
        output_fields = UID->fileOptions->outputFields;

        if (output_fields.empty() || output_fields[0]=="all") {
            output_fields.clear();
            output_fields = {"u","v","w","icell"};
        }

        // set cell-centered dimensions
        NcDim t_dim = output->addDimension("t");
        NcDim z_dim = output->addDimension("z",nz-2);
        NcDim y_dim = output->addDimension("y",ny-1);
        NcDim x_dim = output->addDimension("x",nx-1);

        dim_scalar_t.push_back(t_dim);
        dim_scalar_z.push_back(z_dim);
        dim_scalar_y.push_back(y_dim);
        dim_scalar_x.push_back(x_dim);
        dim_vector.push_back(t_dim);
        dim_vector.push_back(z_dim);
        dim_vector.push_back(y_dim);
        dim_vector.push_back(x_dim);
        dim_vector_2d.push_back(y_dim);
        dim_vector_2d.push_back(x_dim);

        // create attributes
        AttScalarDbl att_t = {&time,  "t", "time",      "s", dim_scalar_t};
        AttVectorDbl att_x = {&x_out, "x", "x-distance", "m", dim_scalar_x};
        AttVectorDbl att_y = {&y_out, "y", "y-distance", "m", dim_scalar_y};
        AttVectorDbl att_z = {&z_out, "z", "z-distance", "m", dim_scalar_z};
        AttVectorDbl att_u = {&u_out, "u", "x-component velocity", "m s-1", dim_vector};
        AttVectorDbl att_v = {&v_out, "v", "y-component velocity", "m s-1", dim_vector};
        AttVectorDbl att_w = {&w_out, "w", "z-component velocity", "m s-1", dim_vector};
        AttVectorDbl att_h = {&terrain,  "terrain", "terrain height", "m", dim_vector_2d};
        AttVectorInt att_i = {&icellflag_out,  "icell", "icell flag value", "--", dim_vector};

        // map the name to attributes
        map_att_scalar_dbl.emplace("t", att_t);
        map_att_vector_dbl.emplace("x", att_x);
        map_att_vector_dbl.emplace("y", att_y);
        map_att_vector_dbl.emplace("z", att_z);
        map_att_vector_dbl.emplace("u", att_u);
        map_att_vector_dbl.emplace("v", att_v);
        map_att_vector_dbl.emplace("w", att_w);
        map_att_vector_dbl.emplace("terrain", att_h);
        map_att_vector_int.emplace("icell", att_i);

        // we will always save time and grid lengths
        output_scalar_dbl.push_back(map_att_scalar_dbl["t"]);
        output_vector_dbl.push_back(map_att_vector_dbl["x"]);
        output_vector_dbl.push_back(map_att_vector_dbl["y"]);
        output_vector_dbl.push_back(map_att_vector_dbl["z"]);
        output_vector_dbl.push_back(map_att_vector_dbl["terrain"]);

        // create list of fields to save
        for (size_t i=0; i<output_fields.size(); i++) {
            std::string key = output_fields[i];
            if (map_att_scalar_dbl.count(key)) {
                output_scalar_dbl.push_back(map_att_scalar_dbl[key]);
            } else if (map_att_vector_dbl.count(key)) {
                output_vector_dbl.push_back(map_att_vector_dbl[key]);
            } else if(map_att_vector_int.count(key)) {
                output_vector_int.push_back(map_att_vector_int[key]);
            }
        }

        // add vector double fields
        for ( AttScalarDbl att : output_scalar_dbl ) {
            output->addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
        }

        // add vector double fields
        for ( AttVectorDbl att : output_vector_dbl ) {
            output->addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
        }

        // add vector int fields
        for ( AttVectorInt att : output_vector_int ) {
            output->addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
        }
    }
}

void Solver::calculateCoefficients(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag,
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


void Solver::defineWalls(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag,
                         float* n, float* m, float* f, float* e, float* h, float* g)

{

	for (int i=1; i<nx-2; i++)
	{
		for (int j=1; j<ny-2; j++)
		{
			for (int k=1; k<nz-2; k++)
			{
				icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
				if (icellflag[icell_cent] !=0) {

					/// Wall below
					if (icellflag[icell_cent-(nx-1)*(ny-1)]==0) {
			    		n[icell_cent] = 0.0;
					}
					/// Wall above
					if (icellflag[icell_cent+(nx-1)*(ny-1)]==0) {
		    			m[icell_cent] = 0.0;
					}
					/// Wall in back
					if (icellflag[icell_cent-1]==0){
						f[icell_cent] = 0.0;
					}
					/// Wall in front
					if (icellflag[icell_cent+1]==0){
						e[icell_cent] = 0.0;
					}
					/// Wall on right
					if (icellflag[icell_cent-(nx-1)]==0){
						h[icell_cent] = 0.0;
					}
					/// Wall on left
					if (icellflag[icell_cent+(nx-1)]==0){
						g[icell_cent] = 0.0;
					}
				}
			}
		}
	}
}


void Solver::getWallIndices(std::vector<int> &icellflag, std::vector<int>& wall_right_indices,
                            std::vector<int>& wall_left_indices,std::vector<int>& wall_above_indices,
                            std::vector<int>& wall_below_indices, std::vector<int>& wall_front_indices,
                            std::vector<int>& wall_back_indices)
{
  for (int i=0; i<nx-1; i++)
  {
    for (int j=0; j<ny-1; j++)
    {
      for (int k=1; k<nz-2; k++)
      {
        icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        icell_face = i + j*nx + k*nx*ny;

        if (icellflag[icell_cent] !=0)
        {

          /// Wall below
          if (icellflag[icell_cent-(nx-1)*(ny-1)]==0)
          {
            wall_below_indices.push_back(icell_face);
            n[icell_cent] = 0.0;
          }
          /// Wall above
          if (icellflag[icell_cent+(nx-1)*(ny-1)]==0)
          {
            wall_above_indices.push_back(icell_face);
            m[icell_cent] = 0.0;
          }
          /// Wall in back
          if (icellflag[icell_cent-1]==0)
          {
            if (i>0)
            {
              wall_back_indices.push_back(icell_face);
              f[icell_cent] = 0.0;
            }
          }
          /// Wall in front
          if (icellflag[icell_cent+1]==0)
          {
            wall_front_indices.push_back(icell_face);
            e[icell_cent] = 0.0;
          }
          /// Wall on right
          if (icellflag[icell_cent-(nx-1)]==0)
          {
            if (j>0)
            {
              wall_right_indices.push_back(icell_face);
              h[icell_cent] = 0.0;
            }
          }
          /// Wall on left
          if (icellflag[icell_cent+(nx-1)]==0)
          {
            wall_left_indices.push_back(icell_face);
            g[icell_cent] = 0.0;
          }
        }
      }
    }
  }

}


void Solver::wallLogBC (std::vector<int>& wall_right_indices,std::vector<int>& wall_left_indices,
                        std::vector<int>& wall_above_indices,std::vector<int>& wall_below_indices,
                        std::vector<int>& wall_front_indices, std::vector<int>& wall_back_indices,
                        double *u0, double *v0, double *w0, float z0)
{
  float ustar_wall;              /**< velocity gradient at the wall */
  float new_ustar;              /**< new ustar value calculated */
  float vel_mag1;               /**< velocity magnitude at the nearest cell to wall in perpendicular direction */
  float vel_mag2;                /**< velocity magnitude at the second cell near wall in perpendicular direction */
  float dist1;                  /**< distance of the center of the nearest cell in perpendicular direction from wall */
  float dist2;                  /**< distance of the center of second near cell in perpendicular direction from wall */
  float wind_dir;               /**< wind direction in parallel planes to wall */

  // Total size of wall indices
  int wall_size = wall_right_indices.size()+wall_left_indices.size()+
                  wall_above_indices.size()+wall_below_indices.size()+
                  wall_front_indices.size()+wall_back_indices.size();

  std::vector<float> ustar;
  ustar.resize(wall_size, 0.0);
  std::vector<int> index;
  index.resize(wall_size, 0.0);
  int j;

  ustar_wall = 0.1;
  wind_dir = 0.0;
  vel_mag1 = 0.0;
  vel_mag2 = 0.0;

  dist1 = 0.5*dz;
  dist2 = 1.5*dz;

  /// apply log law fix to the cells with wall below
  for (size_t i=0; i < wall_below_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[wall_below_indices[i]+nx*ny],u0[wall_below_indices[i]+nx*ny]);
      vel_mag2 = sqrt(pow(u0[wall_below_indices[i]+nx*ny],2.0)+pow(v0[wall_below_indices[i]+nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      w0[wall_below_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_below_indices[i]] = vel_mag1*cos(wind_dir);
      v0[wall_below_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    index[i] = wall_below_indices[i];
    ustar[i] = ustar_wall;
  }

  /// apply log law fix to the cells with wall above
  for (size_t i=0; i < wall_above_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[wall_above_indices[i]-nx*ny],u0[wall_above_indices[i]-nx*ny]);
      vel_mag2 = sqrt(pow(u0[wall_above_indices[i]-nx*ny],2.0)+pow(v0[wall_above_indices[i]-nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      w0[wall_above_indices[i]] = 0;          /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_above_indices[i]] = vel_mag1*cos(wind_dir);
      v0[wall_above_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size();
    index[j] = wall_above_indices[i];
    ustar[j] = ustar_wall;
  }

  dist1 = 0.5*dx;
  dist2 = 1.5*dx;

  /// apply log law fix to the cells with wall in back
  for (size_t i=0; i < wall_back_indices.size(); i++)
  {
    ustar_wall = 0.1;
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_back_indices[i]+1],v0[wall_back_indices[i]+1]);
      vel_mag2 = sqrt(pow(v0[wall_back_indices[i]+1],2.0)+pow(w0[wall_back_indices[i]+1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      u0[wall_back_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      v0[wall_back_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_back_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size();
    index[j] = wall_back_indices[i];
    ustar[j] = ustar_wall;
  }


  /// apply log law fix to the cells with wall in front
  for (size_t i=0; i < wall_front_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_front_indices[i]-1],v0[wall_front_indices[i]-1]);
      vel_mag2 = sqrt(pow(v0[wall_front_indices[i]-1],2.0)+pow(w0[wall_front_indices[i]-1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      u0[wall_front_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      v0[wall_front_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_front_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size()+wall_back_indices.size();
    index[j] = wall_front_indices[i];
    ustar[j] = ustar_wall;
  }


  dist1 = 0.5*dy;
  dist2 = 1.5*dy;

  /// apply log law fix to the cells with wall to right
  for (size_t i=0; i < wall_right_indices.size(); i++)
  {
    ustar_wall = 0.1;          /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_right_indices[i]+nx],u0[wall_right_indices[i]+nx]);
      vel_mag2 = sqrt(pow(u0[wall_right_indices[i]+nx],2.0)+pow(w0[wall_right_indices[i]+nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      v0[wall_right_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_right_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_right_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size()+wall_back_indices.size()+wall_front_indices.size();
    index[j] = wall_right_indices[i];
    ustar[j] = ustar_wall;
  }

  /// apply log law fix to the cells with wall to left
  for (size_t i=0; i < wall_left_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_left_indices[i]-nx],u0[wall_left_indices[i]-nx]);
      vel_mag2 = sqrt(pow(u0[wall_left_indices[i]-nx],2.0)+pow(w0[wall_left_indices[i]-nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      v0[wall_left_indices[i]] = 0;          /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_left_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_left_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size()+wall_back_indices.size()+wall_front_indices.size()+wall_right_indices.size();
    index[j] = wall_left_indices[i];
    ustar[j] = ustar_wall;
  }
}

  void Solver::mergeSort( std::vector<float> &height, std::vector<std::vector<polyVert>> &poly_points, std::vector<float> &base_height, std::vector<float> &building_height)
  {
  	//if the size of the array is 1, it is already sorted
  	if (height.size() == 1)
    {
      return;
    }
  	//make left and right sides of the data
  	std::vector<float> height_L, height_R;
    std::vector<float> base_height_L, base_height_R;
    std::vector<float> building_height_L, building_height_R;
  	std::vector< std::vector <polyVert> > poly_points_L, poly_points_R;
  	height_L.resize(height.size() / 2);
  	height_R.resize(height.size() - height.size() / 2);
    base_height_L.resize(base_height.size() / 2);
  	base_height_R.resize(base_height.size() - base_height.size() / 2);
    building_height_L.resize(building_height.size() / 2);
  	building_height_R.resize(building_height.size() - building_height.size() / 2);
  	poly_points_L.resize(poly_points.size() / 2);
  	poly_points_R.resize(poly_points.size() - poly_points.size() / 2);

  	//copy data from the main data set to the left and right children
  	int lC = 0, rC = 0;
  	for (unsigned int i = 0; i < height.size(); i++)
  	{
  		if (i < height.size() / 2)
  		{
  			height_L[lC] = height[i];
        base_height_L[lC] = base_height[i];
        building_height_L[lC] = building_height[i];
  			poly_points_L[lC++] = poly_points[i];
  		}
  		else
  		{
  			height_R[rC] = height[i];
        base_height_R[rC] = base_height[i];
        building_height_R[rC] = building_height[i];
  			poly_points_R[rC++] = poly_points[i];
  		}
  	}

  	//recursively sort the children
  	mergeSort(height_L, poly_points_L, base_height_L, building_height_L);
  	mergeSort(height_R, poly_points_R, base_height_R, building_height_R);

  	//compare the sorted children to place the data into the main array
  	lC = rC = 0;
  	for (unsigned int i = 0; i < poly_points.size(); i++)
  	{
  		if (rC == height_R.size() || ( lC != height_L.size() &&
  			height_L[lC] > height_R[rC]))
  		{
  			height[i] = height_L[lC];
        base_height[i] = base_height_L[lC];
        building_height[i] = building_height_L[lC];
  			poly_points[i] = poly_points_L[lC++];
  		}
  		else
  		{
  			height[i] = height_R[rC];
        base_height[i] = base_height_R[rC];
        building_height[i] = building_height_R[rC];
  			poly_points[i] = poly_points_R[rC++];
  		}
  	}

  	return;
  }



// Save output at cell-centered values
void Solver :: save(Output* output) {

    // output size and location
    std::vector<size_t> scalar_index;
    std::vector<size_t> scalar_size;
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    std::vector<size_t> vector_index_2d;
    std::vector<size_t> vector_size_2d;

    scalar_index = {static_cast<unsigned long>(output_counter)};
    scalar_size  = {1};
    vector_index = {static_cast<size_t>(output_counter), 0, 0, 0};
    vector_size  = {1, static_cast<unsigned long>(nz-2),static_cast<unsigned long>(ny-1), static_cast<unsigned long>(nx-1)};
    vector_index_2d = {0, 0};
    vector_size_2d  = {static_cast<unsigned long>(ny-1), static_cast<unsigned long>(nx-1)};

    // set time
    time = (double)output_counter;

    // get cell-centered values
    for (int k = 1; k < nz-1; k++){
    	for (int j = 0; j < ny-1; j++){
    		for (int i = 0; i < nx-1; i++){
    			int icell_face = i + j*nx + k*nx*ny;
    			int icell_cent = i + j*(nx-1) + (k-1)*(nx-1)*(ny-1);
    			u_out[icell_cent] = 0.5*(u[icell_face+1]+u[icell_face]);
    			v_out[icell_cent] = 0.5*(v[icell_face+nx]+v[icell_face]);
    			w_out[icell_cent] = 0.5*(w[icell_face+nx*ny]+w[icell_face]);
    			icellflag_out[icell_cent] = icellflag[icell_cent+((nx-1)*(ny-1))];
    		}
    	}
    }

    // loop through 1D fields to save
    for (int i=0; i<output_scalar_dbl.size(); i++) {
        output->saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);
    }

    // loop through 2D double fields to save
    for (int i=0; i<output_vector_dbl.size(); i++) {

        // x,y,z, terrain saved once with no time component
        if (i<3 && output_counter==0) {
            output->saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
        } else if (i==3 && output_counter==0) {
            output->saveField2D(output_vector_dbl[i].name, vector_index_2d,
                                vector_size_2d, *output_vector_dbl[i].data);
        } else {
            output->saveField2D(output_vector_dbl[i].name, vector_index,
                                vector_size, *output_vector_dbl[i].data);
        }
    }

    // loop through 2D int fields to save
    for (int i=0; i<output_vector_int.size(); i++) {
        output->saveField2D(output_vector_int[i].name, vector_index,
                            vector_size, *output_vector_int[i].data);
    }

    // remove x, y, z, terrain from output array after first save
    if (output_counter==0) {
        output_vector_dbl.erase(output_vector_dbl.begin(),output_vector_dbl.begin()+4);
    }
    // increment for next time insertion
    output_counter +=1;
}
