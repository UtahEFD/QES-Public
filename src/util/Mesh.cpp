/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

/**
 * @file Mesh.cpp
 * @brief Serves as a container for a BVH of Triangle objects that
 * represents a connected collection of Triangles.
 *
 * @sa BVH
 * @sa Triangle
 * @sa SphereDirections
 * @sa Ray
 * @sa HitRecord
 */

#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
  return triangleBVH->heightToTri(x, y);
}

void Mesh::calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLength)
{

  // unused: int cellNum = 0;

#pragma acc parallel loop independent
  for (int k = 0; k < dimZ - 1; k++) {
    for (int j = 0; j < dimY - 1; j++) {
      for (int i = 0; i < dimX - 1; i++) {

        // calculate icell index
        int icell_idx = i + j * (dimX - 1) + k * (dimY - 1) * (dimX - 1);

        if (icellflag[icell_idx] == 1) {

          SphereDirections sd(512, -1, 1, 0, 2 * M_PI);

          float maxLength = std::numeric_limits<float>::infinity();

          // ray's origin = cell's center
          Ray ray((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz);

          HitRecord hit;
          //               float t1 = -1;
          //               float t0 = 0;

          // for all possible directions, determine the distance

          for (int m = 0; m < sd.getNumDirVec(); m++) {

            // ray.setDir(sd.getNextDirCardinal());
            ray.setDir(sd.getNextDir());

            bool isHit = triangleBVH->rayHit(ray, hit);

            if (isHit) {
              // std::cout<<"Hit found."<<std::endl;

              // compare the mixLengths
              if (hit.hitDist < maxLength) {
                maxLength = hit.hitDist;
                // std::cout<<"maxlength updated"<<std::endl;
              }
            } else {
              // std::cout<<"Hit not found"<<std::endl;
              // std::cout<<"Hit may not be found but hit.hitDist = "<<hit.hitDist<<std::endl;
            }
          }

          // std::cout<<"Mixing length for this cell is "<<maxLength<<std::endl;
          // add to list of vectors
          mixingLength[icell_idx] = maxLength;
          // std::cout<<"\n\n"<<std::endl;
        }
      }
    }
  }
}


// This needs to be removed from here in next round of edits.  Just
// marking now.
void Mesh::tempOPTIXMethod(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths)
{
  std::cout << "--------------Enters the tempOPTIXMethod--------------------" << std::endl;
  //   OptixRayTrace optixRayTracer(optixTris);
  //   optixRayTracer.calculateMixingLength( mlSampleRate, dimX, dimY, dimZ, dx, dy, dz, icellflag, mixingLengths);
}
