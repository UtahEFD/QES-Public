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

#include <numeric>

#include "WallReflection_StairStep.h"

void WallReflection_StairStep::reflect(const WINDSGeneralData *WGD,
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
    state = ACTIVE;
  } else {
    // particle end trajectory inside solide -> need for reflection

    // position of the particle start of trajectory
    Vector3Float X = { pos._1 - dist._1, pos._2 - dist._2, pos._3 - dist._3 };
    // vector of the trajectory
    Vector3Float U = { dist._1, dist._2, dist._3 };
    // postion of the particle end of trajectory
    Vector3Float Xnew = X + U;
    // vector of fluctuations
    Vector3Float vecFluct = { fluct._1, fluct._2, fluct._3 };

    float d = U.length();
    float d1 = 0.0;
    float d2 = U.length();

    // normailzation of direction verctor for particle trajectory
    U = U / U.length();
    bool isActive = true;
    trajectorySplit_recursive(WGD, X, U, d, d1, d2, vecFluct, isActive);

    if (isActive) {
      // update output variable: particle position
      pos._1 = X[0];
      pos._2 = X[1];
      pos._3 = X[2];
      // update output variable: fluctuations
      fluct._1 = vecFluct[0];
      fluct._2 = vecFluct[1];
      fluct._3 = vecFluct[2];

      state = ACTIVE;
    } else {
      state = INACTIVE;
    }
  }
  // should never be there
}

void WallReflection_StairStep::trajectorySplit_recursive(const WINDSGeneralData *WGD,
                                                         Vector3Float &X,
                                                         Vector3Float &u,
                                                         const float &d,
                                                         float &d1,
                                                         float &d2,
                                                         Vector3Float &vecFluct,
                                                         bool &isActive)
{
  /*
    - d total distance
    - d1 total distance traveled so far
    - d2 distance traveled this step
  */

  if (std::abs(d1 - d) < 1.0E-3) {
    // std::cerr << "END OF REGRESSION dist \t" << std::abs(d1 - d) << std::endl;
    //  isActive = true;
    return;
  } else if (!isActive) {
    return;
  } else {

    Vector3Float Xold = X;
    Vector3Float Xnew = X + d2 * u;

    // linearized cell ID for origine of the trajectory of the particle
    long cellIdOld = m_interp->getCellId(Xold);
    // i,j,k of cell index
    auto [i_old, j_old, k_old] = m_interp->getCellIndex(cellIdOld);

    // linearized cell ID for origine of the trajectory of the particle
    long cellIdNew = m_interp->getCellId(Xnew);
    // i,j,k of cell index
    auto [i_new, j_new, k_new] = m_interp->getCellIndex(cellIdNew);


    if ((abs(i_old - i_new) > 1) || (abs(j_old - j_new) > 1) || (abs(k_old - k_new) > 1)) {
      d2 = 0.5f * d2;
      if (d2 < 0.125 * d) {
        std::cerr << "END OF REGRESSION dist \t" << d2 / d << std::endl;
        isActive = false;
      }

    } else {
      d1 += d2;
      // X = Xold + d2 * u;
      oneReflection(WGD, X, u, d2, vecFluct, isActive);
    }
    // regression
    trajectorySplit_recursive(WGD, X, u, d, d1, d2, vecFluct, isActive);

    // isActive = true;
    return;
  }
  return;
}

void WallReflection_StairStep::oneReflection(const WINDSGeneralData *WGD,
                                             Vector3Float &X,
                                             Vector3Float &u,
                                             const float &d,
                                             Vector3Float &vecFluct,
                                             bool &isActive)
{
  // some constants
  const float eps_S = 0.001;
  const int maxCount = 10;

  // QES-winds grid information
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // cartesian basis vectors
  const Vector3Float e1 = { 1.0, 0.0, 0.0 }, e2 = { 0.0, 1.0, 0.0 }, e3 = { 0.0, 0.0, 1.0 };

  /* Vector3 variables informations:
     Xold     = origine of the trajectory of the particle
     Xnew     = end of the trajectory of the particle
     vecFluct = fluctuation of the particle
     P        = position of the particle on the wall where bounce happens
     S        = location of the wall
     U        = trajectory of the particle
     V1       = trajectory of the particle to the wall V1 = P - Xold
     V2       = trajectory of the particle to the wall V1 = Xnew - P
     R        = unit vector giving orentation of the reflection
     N        = unit vector noraml to the surface
  */
  Vector3Float Xnew, Xold, U;
  Vector3Float P, S, V1, V2;
  Vector3Float R, N;

  Xold = X;
  U = d * u;
  Xnew = Xold + U;

  // i,j,k of cell index
  long cellIdOld = m_interp->getCellId(Xold);
  long cellIdNew = m_interp->getCellId(Xnew);

  // icellFlag of the cell at the end of the trajectory of the particle
  int cellFlagNew = WGD->icellflag.at(cellIdNew);


  /* Working variables informations:
     count       - number of reflections
     f1,f2,f3    - sign of trajectory in each direction (+/-1)
     l1,l2,l3    - ratio of distance to wall over total distance travel to closest surface in
     -             each direction: by definition positive, if < 1 -> reflection possible
     -             if > 1 -> surface too far
     validSuface - number of potential valid surface
     s           - smallest ratio of dist. to wall over total dist. travel (once surface selected)
     r           - distance travel after reflection
  */
  int count = 0;
  float f1, f2, f3;
  float l1, l2, l3;
  int validSurface;
  float s, r;

  while ((cellFlagNew == 0 || cellFlagNew == 2) && (count < maxCount)) {

    cellIdOld = m_interp->getCellId(Xold);
    auto [i, j, k] = m_interp->getCellIndex(cellIdOld);

    // set direction
    f1 = (e1 * U);
    f1 = f1 / std::abs(f1);
    f2 = (e2 * U);
    f2 = f2 / std::abs(f2);
    f3 = (e3 * U);
    f3 = f3 / std::abs(f3);

    // reset smallest ratio
    s = 100.0;
    // reset number of potential valid surface
    validSurface = 0;
    // reset distance travel after all
    r = 0.0;

    // x-drection
    N = -f1 * e1;
    S = { WGD->domain.x[i] + f1 * 0.50f * dx, WGD->domain.y[j], WGD->domain.z[k] };
    l1 = -(Xold * N - S * N) / (U * N);

    // y-drection
    N = -f2 * e2;
    S = { WGD->domain.x[i], WGD->domain.y[j] + f2 * 0.50f * dy, WGD->domain.z[k] };
    l2 = -(Xold * N - S * N) / (U * N);

    // z-drection (dz can be variable with hieght)
    N = -f3 * e3;
    if (f3 >= 0.0) {
      S = { WGD->domain.x[i], WGD->domain.y[j], WGD->domain.z_face[k + 1] };
    } else {
      S = { WGD->domain.x[i], WGD->domain.y[j], WGD->domain.z_face[k] };
    }
    l3 = -(Xold * N - S * N) / (U * N);

    // check with surface is a potential bounce (0 < li < 1.0)
    // if li=1 -> particle in surface, -> cause problem as particle will stay
    // on the surface, move the reflection point slightly.
    // check for surface in the x-direction
    if ((l1 >= -eps_S) && (l1 <= 1.0 - eps_S)) {
      validSurface++;
      s = l1;
      N = -f1 * e1;
    } else if ((l1 >= 1.0 - eps_S) && (l1 <= 1.0 + eps_S)) {
      validSurface++;
      l1 -= 2 * eps_S;
      s = l1;
      N = -f1 * e1;
    }
    // check for surface in the y-direction
    if ((l2 >= -eps_S) && (l2 <= 1.0 - eps_S)) {
      validSurface++;
      s = l2;
      N = -f2 * e2;
    } else if ((l2 >= 1.0 - eps_S) && (l2 <= 1.0 + eps_S)) {
      validSurface++;
      l2 -= 2 * eps_S;
      s = l2;
      N = -f2 * e2;
    }
    // check for surface in the z-direction
    if ((l3 >= -eps_S) && (l3 <= 1.0 - eps_S)) {
      validSurface++;
      s = l3;
      N = -f3 * e3;
    } else if ((l3 >= 1.0 - eps_S) && (l3 <= 1.0 + eps_S)) {
      validSurface++;
      l3 -= 2 * eps_S;
      s = l3;
      N = -f3 * e3;
    }

    // check if more than one surface is valid
    if (validSurface == 0) {
      // if 0 valid surface
      std::cerr << "[WARNING]\tReflection problem: no valid surface" << std::endl;
      // std::cerr << "\tReflection problem: no valid surface\n"
      //           << "\t" << count << " " << U.length() << "->"
      //           << "[" << l1 << "," << l2 << "," << l3 << "]"
      //           << std::endl;
      // exit(EXIT_FAILURE);
      isActive = false;
      return;
    } else if (validSurface == 1) {
      // only one surface -> s and N already set above
      /* NOTE: the particle travel between fluid -> solid, if only one surface is valid
       *       the surface detected above has to be the reflection surface.
       */

      // for debug
      // std::cerr << "[x,x,x][" << l1 << "," << l2 << "," << l3 << "] " << std::endl;
    } else if (validSurface > 1) {
      // Here-> Multiple options to bounce
      // need to find the best surface
      /* NOTE: the particle travel between fluid -> solid, if only multiple surface is valid
       *       that means that the particle travel across a face in more that one direction.
       *       Some cell might be fluid, at least one will be solid. Need to check icellflag.
       */

      // list of potential surface
      // - ratio of dist. to wall over dist. total
      std::vector<float> vl(3, 0.0);
      // - normal vector for each surface
      std::vector<Vector3Float> vN(3, { 0, 0, 0 });
      // - linear index for icellflag check
      std::vector<int> vn(3, 0);

      // add potential surface to the list (valid only if 0 <= li <= 1.0)
      // -> only executed if 2 valid surfaces exist
      // surface in the x-direction
      vl[0] = l1;
      vN[0] = -f1 * e1;
      vn[0] = f1;
      // surface in the y-direction
      vl[1] = l2;
      vN[1] = -f2 * e2;
      vn[1] = f2 * (nx - 1);
      // surface in the z-direction
      vl[2] = l3;
      vN[2] = -f3 * e3;
      vn[2] = f3 * (nx - 1) * (ny - 1);

      // sort indices from smallest to largest (only indices are sorted)
      std::vector<size_t> idx(vl.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(), [&vl](size_t i1, size_t i2) { return vl[i1] < vl[i2]; });

      // for debug
      // std::cerr << "[" << idx[0] << "," << idx[1] << "," << idx[2] << "]"
      //          << "[" << vl[0] << "," << vl[1] << "," << vl[2] << "] " << std::endl;

      // check if surface is valid (ie, next cell is solid)
      if ((WGD->icellflag.at(cellIdOld + vn[idx[0]]) == 0)
          || (WGD->icellflag.at(cellIdOld + vn[idx[0]]) == 2)) {
        s = vl[idx[0]];
        N = vN[idx[0]];
      } else if ((WGD->icellflag.at(cellIdOld + vn[idx[0]] + vn[idx[1]]) == 0)
                 || (WGD->icellflag.at(cellIdOld + vn[idx[0]] + vn[idx[1]]) == 2)) {
        s = vl[idx[1]];
        N = vN[idx[1]];
      } else if ((WGD->icellflag.at(cellIdOld + vn[idx[0]] + vn[idx[1]] + vn[idx[2]]) == 0)
                 || (WGD->icellflag.at(cellIdOld + vn[idx[0]] + vn[idx[1]] + vn[idx[2]]) == 2)) {
        s = vl[idx[2]];
        N = vN[idx[2]];
      } else {
        // this should happend only if particle traj. more than 1 cell in each direction,
        // -> should have been skipped at the beginning of the function
        // std::cerr << "Reflection problem: no valid surface" << std::endl;
        isActive = false;
        return;
      }
    }
    // no else -> only one valid surface

    // vector from current postition to the wall
    V1 = s * U;
    // postion of reflection on the wall
    P = Xold + V1;
    // distance traveled after the wall
    V2 = U - V1;
    r = V2.length();
    // reflection: normalizing V2 -> R is of norm 1
    V2 = V2 / V2.length();
    R = V2.reflect(N);
    // update postion from surface reflection
    Xnew = P + r * R;
    // reflection of the Fluctuation
    vecFluct = vecFluct.reflect(N);
    // vecFluct_old = vecFluct_old.reflect(N);
    //  prepare variables for next bounce: particle position
    Xold = P;

    // distance travelled by particle
    U = Xnew - Xold;

    // increment the reflection count
    count = count + 1;
    // update the icellflag
    cellIdNew = m_interp->getCellId(Xnew);

    try {
      cellFlagNew = WGD->icellflag.at(cellIdNew);
    } catch (const std::out_of_range &oor) {
      // cell ID out of bound
      // std::cerr << "Reflection problem: particle out of range after reflection" << std::endl;
      isActive = false;
      return;
    }

  }// end of: while( (cellFlagNew==0 || cellFlagNew==2) && (count < maxCount) )

  if (count < maxCount) {
    X = Xnew;
    u = U / U.length();
    isActive = true;
    return;
  } else {
    X = Xold;
    isActive = false;
    return;
  }
}
