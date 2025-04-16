/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file WallReflection_StairStep.cpp
 * @brief
 */

#include "WallReflection_TriMesh.h"

void WallReflection_TriMesh::reflect(const WINDSGeneralData *WGD,
                                     vec3 &pos,
                                     vec3 &dist,
                                     vec3 &fluct,
                                     ParticleState &state)
{
  /*
   * This function will return true if:
   * - no need for reflection (leave xPos, yPos, zPos, uFluct, vFluct, wFluct unchanged)
   * - reflection valid
   * - reflection invalid
   *   if: - count > maxCount
   *       - particle trajectory more than 1 cell in each direction
   *   => in this case, the particle will be place back ot the old position, new random move next step.
   *
   *
   * This function will return false (leave xPos, yPos, zPos, uFluct, vFluct, wFluct unchanged)
   * - cell ID out of bound
   * - no valid surface for reflection
   */


  // linearized cell ID for end of the trajectory of the particle
  long cellIdNew = m_interp->getCellId(pos);
  int cellFlag(0);
  try {
    cellFlag = WGD->icellflag.at(cellIdNew);
  } catch (const std::out_of_range &oor) {
    // cell ID out of bound (assuming particle outside of domain)
    if (pos._3 < m_interp->getZstart()) {
      // assume in terrain icellflag
      cellFlag = 2;
    } else {
      // otherwise, outside domain -> set to false
      // std::cerr << "Reflection problem: particle out of range before reflection" << std::endl;
      state = INACTIVE;
    }
  }

  if ((cellFlag != 0) && (cellFlag != 2)) {
    // particle end trajectory outside solide -> no need for reflection
  } else {

    Vector3Float X = { pos._1 - dist._1, pos._2 - dist._2, pos._3 - dist._3 };

    // vector of the trajectory
    Vector3Float U = { dist._1, dist._2, dist._3 };
    // postion of the particle end of trajectory
    Vector3Float Xnew = X + U;
    // vector of fluctuations
    Vector3Float vecFluct = { fluct._1, fluct._2, fluct._3 };

    rayTraceReflect(WGD->mesh, X, Xnew, U, vecFluct);

    pos._1 = Xnew[0];
    pos._2 = Xnew[1];
    pos._3 = Xnew[2];
    // update output variable: fluctuations
    fluct._1 = vecFluct[0];
    fluct._2 = vecFluct[1];
    fluct._3 = vecFluct[2];
  }
  state = ACTIVE;
}

void WallReflection_TriMesh::rayTraceReflect(Mesh *mesh, Vector3Float &X, Vector3Float &Xnew, Vector3Float &U, Vector3Float &vecFluct)
{
  Ray test_ray(X, U);
  HitRecord hit;
  if (mesh->triangleBVH->rayHit(test_ray, hit)) {
    if (hit.getHitDist() <= U.length()) {

      // std::cout << "----\n";
      // std::cout << "hit the mesh at " << hit.endpt << " " << X << " " << Xnew << "\n";
      // std::cout << "A\thit dist " << hit.getHitDist() << "/" << U.length() << "=" << hit.getHitDist() / U.length() << std::endl;
      // std::cout << "hit normal " << hit.n << std::endl;

      Vector3Float P, S, V2;
      Vector3Float R, N;
      float r;

      N = hit.n;

      // postion of reflection on the wall
      P = hit.endpt - 0.1 * U / U.length();
      // distance traveled after the wall
      V2 = Xnew - P;
      r = V2.length();
      // reflection: normalizing V2 -> R is of norm 1
      V2 = V2 / V2.length();
      R = V2.reflect(N);
      // update postion from surface reflection
      Xnew = P + r * R;
      // reflection of the Fluctuation
      vecFluct = vecFluct.reflect(N);

      // prepare variables for next bounce: particle position
      X = P;
      // prepare variables for next bounce: distance left to be travelled by particle
      U = Xnew - X;

      rayTraceReflect(mesh, X, Xnew, U, vecFluct);
    } else {
      // hit too far
      // std::cout << "B\thit dist " << hit.getHitDist() << "/" << U.length() << "=" << hit.getHitDist() / U.length() << std::endl;
    }
  } else {
    // no hit
  }
}
