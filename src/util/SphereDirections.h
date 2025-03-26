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

/** @file SphereDirections.h */

#pragma once

#include <cmath>
#include <cfloat>
#include <random>

#include "Vector3Float.h"

/**
 * @class SphereDirections
 * @brief Used to generate direction vectors for a sphere to use in ray tracing.
 */
class SphereDirections
{
private:
  int vecCount; /**< :document this: */
  int numDirs; /**< :document this: */

  ///@{
  /** Range of sphere for random version. */
  float lowerThetaBound;
  float upperThetaBound;
  float lowerPhiBound;
  float upperPhiBound;
  ///@}

  // std::vector< Vector3Float > nextList; // [6];  //holds vectors of the 6

  /**
   * Constructor for the Mitchell's Best Candidate Algorithm test.
   */
  SphereDirections(int numDirVec)
    : vecCount(0), numDirs(numDirVec),
      lowerThetaBound(0.0), upperThetaBound(0.0),
      lowerPhiBound(0.0), upperPhiBound(0.0)
  {
    nextList = new Vector3Float[6];

    // default cardinal directions for now
    nextList[0] = Vector3Float(1, 0, 0);// front
    nextList[1] = Vector3Float(-1, 0, 0);// back
    nextList[2] = Vector3Float(0, 1, 0);// left
    nextList[3] = Vector3Float(0, -1, 0);// right
    nextList[4] = Vector3Float(0, 0, 1);// top
    nextList[5] = Vector3Float(0, 0, -1);// bottom
  }

  /**
   * Default constuctor for the 6 cardinal directions.
   */
  SphereDirections()
    : vecCount(0),
      lowerThetaBound(0.0), upperThetaBound(0.0),
      lowerPhiBound(0.0), upperPhiBound(0.0)
  {
    nextList = new Vector3Float[6];

    // default cardinal directions for now
    nextList[0] = Vector3Float(1, 0, 0);// front
    nextList[1] = Vector3Float(-1, 0, 0);// back
    nextList[2] = Vector3Float(0, 1, 0);// left
    nextList[3] = Vector3Float(0, -1, 0);// right
    nextList[4] = Vector3Float(0, 0, 1);// top
    nextList[5] = Vector3Float(0, 0, -1);// bottom

    numDirs = 6;
  }

  Vector3Float *nextList; /**< :document this: */

public:
  /**
   * Constuctor for the random version.
   *
   * @param numDV :document this:
   * @param lowerTheta :document this:
   * @param upperTheta :document this:
   * @param lowerPhi :document this:
   * @param upperPhi :document this:
   */
  SphereDirections(int numDV, float lowerTheta, float upperTheta, float lowerPhi, float upperPhi)
    : vecCount(0),
      lowerThetaBound(lowerTheta), upperThetaBound(upperTheta),
      lowerPhiBound(lowerPhi), upperPhiBound(upperPhi)

  {
    //        std::random_device rd;  // the rd device reads from a file,
    //        apparently and thus, calls strlen, might need some other way
    //        to seed.

    //        std::mt19937 e2(rd());
    std::mt19937 e2(303);
    std::uniform_real_distribution<float> theta(lowerThetaBound, upperThetaBound);
    std::uniform_real_distribution<float> phi(lowerPhiBound, upperPhiBound);

    numDirs = numDV + 5;
    nextList = new Vector3Float[numDirs];

    // for (int i=0; i<numDV; i++) {
    int i = 0;
    while (i < numDV) {

      float theta2 = std::asin(theta(e2));

      float dx = std::cos(theta2) * std::cos(phi(e2));
      float dy = std::sin(phi(e2));
      float dz = std::cos(theta2) * std::sin(phi(e2));

      float magnitude = std::sqrt(dx * dx + dy * dy + dz * dz);

      /* FM CLEANUP - NOT USED
      // only send rays mostly down but a little up... can use
      // dot product between (0, 0, 1) and vector
      Vector3Float dirVec(dx / magnitude, dy / magnitude, dz / magnitude);
      float dotProd = dirVec[0] * 0.0f + dirVec[1] * 0.0f + dirVec[2] * 1.0f;
      */

      // if (dotProd < 0.20) {
      nextList[i] = Vector3Float(dx / magnitude, dy / magnitude, dz / magnitude);
      i++;
      // }
    }

    // Then make sure the cardinal directions that may matter are
    // added -- up is unlikely at this point
    nextList[numDV] = Vector3Float(1, 0, 0);// front
    nextList[numDV + 1] = Vector3Float(-1, 0, 0);// back
    nextList[numDV + 2] = Vector3Float(0, 1, 0);// left
    nextList[numDV + 3] = Vector3Float(0, -1, 0);// right
    nextList[numDV + 4] = Vector3Float(0, 0, -1);// bottom

    //        std::cout << "Generated " << nextList.size() << " sphere directions." << std::endl;
    //        std::cout << "sd = [" << std::endl;
    //        for (int i=0; i<nextList.size(); i++) {
    //            std::cout << "\t" << nextList[i][0] << " " << nextList[i][1] << " " << nextList[i][2] << ";" << std::endl;
    //        }
    //        std::cout << "];" << std::endl;
  }


  ~SphereDirections()
  {
    delete[] nextList;
  }


  /**
   * @return numDirVec the number of directional vectors generated
   */
  int getNumDirVec() { return numDirs; }

  /*
   * @return the next cardinal directional vector or NULL if the vecCount > numDirVec
   */
  //   Vector3Float getNextDirCardinal();

  /**
   * Gets a randomly generated directional vector based on theta and
   * phi bounds.
   *
   * @return the next randomly generated directional vector
   */
  //   Vector3Float getNextDir();
  Vector3Float getNextDir()
  {
    Vector3Float retVal = nextList[vecCount];

    vecCount++;
    if (vecCount > numDirs)
      vecCount = numDirs - 1;

    return retVal;
  }


  /*Mitchel's Best Algorithm
   *Gets the next unique direction
   *@return the next non-repeated directional vector
   */
  //   Vector3Float getNextDir2();
};
