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

/** @file WallReflection.cpp 
 * @brief These functions handle the different wall reflection options
 *
 * @note Part of plume class 
 */

#include "Plume.hpp"

// reflection -> set particle inactive when entering a wall
bool Plume::wallReflectionSetToInactive(const WINDSGeneralData *WGD,
                                        double &xPos,
                                        double &yPos,
                                        double &zPos,
                                        double &disX,
                                        double &disY,
                                        double &disZ,
                                        double &uFluct,
                                        double &vFluct,
                                        double &wFluct)
{
  try {
    int cellIdx = interp->getCellId(xPos, yPos, zPos);
    int cellFlag(0);
    cellFlag = WGD->icellflag.at(cellIdx);

    if ((cellFlag == 0) || (cellFlag == 2)) {
      // particle end trajectory inside solide -> set inactive
      return false;
    } else {
      // particle end trajectory outside solide -> keep active
      return true;
    }

  } catch (const std::out_of_range &oor) {
    // cell ID out of bound (assuming particle outside of domain)
    if (zPos < domainZstart) {
      std::cerr << "Reflection problem: particle out of range before reflection" << std::endl;
      std::cerr << xPos << "," << yPos << "," << zPos << std::endl;
    }
    // -> set to false
    return false;
  }
}

// reflection -> this function will do nothing
bool Plume::wallReflectionDoNothing(const WINDSGeneralData *WGD,
                                    double &xPos,
                                    double &yPos,
                                    double &zPos,
                                    double &disX,
                                    double &disY,
                                    double &disZ,
                                    double &uFluct,
                                    double &vFluct,
                                    double &wFluct)
{
  return true;
}

bool Plume::wallReflectionFullStairStep(const WINDSGeneralData *WGD,
                                        double &xPos,
                                        double &yPos,
                                        double &zPos,
                                        double &disX,
                                        double &disY,
                                        double &disZ,
                                        double &uFluct,
                                        double &vFluct,
                                        double &wFluct)
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
  int cellIdNew = interp->getCellId(xPos, yPos, zPos);
  int cellFlag(0);
  try {
    cellFlag = WGD->icellflag.at(cellIdNew);
  } catch (const std::out_of_range &oor) {
    // cell ID out of bound (assuming particle outside of domain)
    if (zPos < domainZstart) {
      std::cerr << "Reflection problem: particle out of range before reflection" << std::endl;

      Vector3Double Xnew, Xold, U, vecFluct;
      // position of the particle start of trajectory
      Xold = { xPos - disX, yPos - disY, zPos - disZ };
      // postion of the particle end of trajectory
      Xnew = { xPos, yPos, zPos };
      U = Xnew - Xold;
      vecFluct = { uFluct, vFluct, wFluct };

      std::cerr << "Xnew\t" << Xnew << std::endl;
      std::cerr << "Xold\t" << Xold << std::endl;
      std::cerr << "U\t" << U << " " << U.length() << std::endl;
    }
    // -> set to false
    return false;
  }

  if ((cellFlag != 0) && (cellFlag != 2)) {
    // particle end trajectory outside solide -> no need for reflection
    return true;
  }

  // linearized cell ID for origine of the trajectory of the particle
  int cellIdOld = interp->getCellId(xPos - disX, yPos - disY, zPos - disZ);

  // i,j,k of cell index
  Vector3Int cellIdxOld = interp->getCellIndex(cellIdOld);
  Vector3Int cellIdxNew = interp->getCellIndex(cellIdNew);
  int i = cellIdxOld[0], j = cellIdxOld[1], k = cellIdxOld[2];

  // check particle trajectory more than 1 cell in each direction
  if ((abs(cellIdxOld[0] - cellIdxNew[0]) > 1)
      || (abs(cellIdxOld[1] - cellIdxNew[1]) > 1)
      || (abs(cellIdxOld[2] - cellIdxNew[2]) > 1)) {
    // update output variable: particle position to old position
    xPos -= disX;
    yPos -= disY;
    zPos -= disZ;

    // for debug
    std::cerr << "Reflection problem: particle moved too fast: cell traveled: "
              << abs(cellIdxOld[0] - cellIdxNew[0]) << ","
              << abs(cellIdxOld[1] - cellIdxNew[1]) << ","
              << abs(cellIdxOld[2] - cellIdxNew[2]) << std::endl;

    Vector3Double Xnew, Xold, U, vecFluct;
    // position of the particle start of trajectory
    Xold = { xPos - disX, yPos - disY, zPos - disZ };
    // postion of the particle end of trajectory
    Xnew = { xPos, yPos, zPos };
    U = Xnew - Xold;
    vecFluct = { uFluct, vFluct, wFluct };

    std::cerr << "Xnew\t" << Xnew << std::endl;
    std::cerr << "Xold\t" << Xold << std::endl;
    std::cerr << "U\t" << U << " " << U.length() << std::endl;
    //std::cerr << "u\t" << vecFluct << " " << vecFluct.length() << std::endl;
    Vector3Double X = 0.5 * U;
    double d = X.length();
    std::cerr << "dist\t" << d << std::endl;
    X = Xold + X;
    std::cerr << "X\t" << X << std::endl;

    X = X + U / U.length() * d;

    std::cerr << "X\t" << X << std::endl;

    return true;
  }

  // some constants
  const double eps_S = 0.001;
  const int maxCount = 10;

  // QES-winds grid information
  int nx = WGD->nx;
  int ny = WGD->ny;
  //int nz = WGD->nz;
  double dx = WGD->dx;
  double dy = WGD->dy;
  //double dz = WGD->dz;

  // cartesian basis vectors
  const Vector3Double e1 = { 1.0, 0.0, 0.0 }, e2 = { 0.0, 1.0, 0.0 }, e3 = { 0.0, 0.0, 1.0 };

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
  Vector3Double Xnew, Xold;
  Vector3Double vecFluct, vecFluct_old;
  Vector3Double P, S, U, V1, V2;
  Vector3Double R, N;

  // position of the particle start of trajectory
  Xold = { xPos - disX, yPos - disY, zPos - disZ };

  // postion of the particle end of trajectory
  Xnew = { xPos, yPos, zPos };

  // icellFlag of the cell at the end of the trajectory of the particle
  int cellFlagNew = WGD->icellflag.at(cellIdNew);

  // vector of fluctuation
  vecFluct = { uFluct, vFluct, wFluct };
  //vecFluct_old = { uFluct_old, vFluct_old, wFluct_old };

  /* Working variables informations:
     count       - number of reflections
     f1,f2,f3    - sign of trajectory in each direction (+/-1)
     l1,l2,l3    - ratio of distance to wall over total distance travel to closest surface in 
     -             each direction: by definition positive, if < 1 -> reflection possible
     -             if > 1 -> surface too far
     validSuface - number of potential valid surface
     s           - smallest ratio of dist. to wall over total dist. travel (once surface selected)
     d           - distance travel after reflection
  */
  int count = 0;
  double f1, f2, f3;
  double l1, l2, l3;
  int validSurface;
  double s, d;

  while ((cellFlagNew == 0 || cellFlagNew == 2) && (count < maxCount)) {

    // distance travelled by particle
    U = Xnew - Xold;

    //set direction
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
    d = 0.0;

    // x-drection
    N = -f1 * e1;
    S = { WGD->x[i] + f1 * 0.50 * dx, double(WGD->y[j]), double(WGD->z[k]) };
    l1 = -(Xold * N - S * N) / (U * N);

    // y-drection
    N = -f2 * e2;
    S = { double(WGD->x[i]), WGD->y[j] + f2 * 0.50 * dy, double(WGD->z[k]) };
    l2 = -(Xold * N - S * N) / (U * N);

    // z-drection (dz can be variable with hieght)
    N = -f3 * e3;
    if (f3 >= 0.0) {
      S = { double(WGD->x[i]), double(WGD->y[j]), double(WGD->z_face[k]) };
    } else {
      S = { double(WGD->x[i]), double(WGD->y[j]), double(WGD->z_face[k - 1]) };
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
      std::cerr << "\tReflection problem: no valid surface\n"
                << "\t" << count << " " << U.length() << "->"
                << "[" << l1 << "," << l2 << "," << l3 << "]" << std::endl;
      //exit(EXIT_FAILURE);
      return false;
    } else if (validSurface == 1) {
      // only one surface -> s and N already set above
      /* NOTE: the particle travel between fluid -> solid, if only one surface is valid
       *       the surface detected above has to be the reflection surface.
       */

      // for debug
      //std::cerr << "[x,x,x][" << l1 << "," << l2 << "," << l3 << "] " << std::endl;
    } else if (validSurface > 1) {
      // Here-> Multiple options to bounce
      // need to find the best surface
      /* NOTE: the particle travel between fluid -> solid, if only multiple surface is valid
       *       that means that the particle travel across a face in more that one direction. 
       *       Some cell might be fluid, at least one will be solid. Need to chech icellflag.
       */

      // list of potential surface
      // - ratio of dist. to wall over dist. total
      std::vector<double> vl(3, 0.0);
      // - normal vector for each surface
      std::vector<Vector3Double> vN(3, { 0, 0, 0 });
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
      //std::cerr << "[" << idx[0] << "," << idx[1] << "," << idx[2] << "]"
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
        //std::cout << "Reflection problem: no valid surface" << std::endl;
        return true;
      }
    }
    // no else -> only one valid surface

    // vector from current postition to the wall
    V1 = s * U;
    // postion of relfection on the wall
    P = Xold + V1;
    // distance traveled after the wall
    V2 = U - V1;
    d = V2.length();
    // reflection: normalizing V2 -> R is of norm 1
    V2 = V2 / V2.length();
    R = V2.reflect(N);
    // update postion from surface reflection
    Xnew = P + d * R;
    // relfection of the Fluctuation
    vecFluct = vecFluct.reflect(N);
    //vecFluct_old = vecFluct_old.reflect(N);
    // prepare variables for next bounce: particle position
    Xold = P;

    // increment the relfection count
    count = count + 1;
    // update the icellflag
    cellIdNew = interp->getCellId(Xnew);

    try {
      cellFlagNew = WGD->icellflag.at(cellIdNew);
    } catch (const std::out_of_range &oor) {
      // cell ID out of bound
      std::cout << "Reflection problem: particle out of range after reflection" << std::endl;
      std::cerr << xPos << "," << yPos << "," << zPos << std::endl;
      return false;
    }

  }// end of: while( (cellFlagNew==0 || cellFlagNew==2) && (count < maxCount) )

  //std::cout << Xold << " " << Xnew << std::endl;

  if (count < maxCount) {
    // update output variable: particle position
    xPos = Xnew[0];
    yPos = Xnew[1];
    zPos = Xnew[2];
    // update output variable: fluctuations
    uFluct = vecFluct[0];
    vFluct = vecFluct[1];
    wFluct = vecFluct[2];
    // update output variable: old fluctuations
    //uFluct_old = vecFluct_old[0];
    //vFluct_old = vecFluct_old[1];
    //wFluct_old = vecFluct_old[2];
  } else {
    // update output variable: particle position to old position
    xPos -= disX;
    yPos -= disY;
    zPos -= disZ;
  }

  return true;
}
