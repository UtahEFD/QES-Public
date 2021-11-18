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

/** @file StreetIntersection */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


/**
 *
 * This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
 * This function reads in building features like nodes, building height and base height and uses
 * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
 * cells qualified on top of buildings and applies the approperiate parameterization to them.
 * More information:
 *
 */
/*void PolyBuilding::streetIntersection (const WINDSInputData* WID, WINDSGeneralData* WGD)
{

  int NS_flag;
  int change_flag = 0;
  int i_start_flag, j_start_flag;
  std::vector<int> intersect, intersect_1, intersect_2;
  std::vector<int> intersect_1opp, intersect_2opp;
  std::vector<int> E_W_flag, W_E_flag, N_S_flag, S_N_flag;

  intersect.resize (WGD->numcell_cent, 0);
  intersect_1.resize (WGD->numcell_cent, 0);
  intersect_2.resize (WGD->numcell_cent, 0);
  intersect_1opp.resize (WGD->numcell_cent, 0);
  intersect_2opp.resize (WGD->numcell_cent, 0);

  E_W_flag.resize (WGD->numcell_cent, 0);
  W_E_flag.resize (WGD->numcell_cent, 0);
  N_S_flag.resize (WGD->numcell_cent, 0);
  S_N_flag.resize (WGD->numcell_cent, 0);

  WGD->wall->setVelocityZero (WGD);

  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto j = 0; j < WGD->ny-1; j++)
    {
      for (auto i = 1; i < WGD->nx-1; i++)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (WGD->icellflag[icell_cent-1] == 6 && WGD->icellflag[icell_cent] != 6 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
        {
          change_flag = 1;
          i_start_flag = i;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        }
        if ((change_flag == 1 && WGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && WGD->icellflag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 0;
        }
        intersect_1[icell_cent] = change_flag;
      }

      //std::cout << "i_start_flag:  " << i_start_flag << std::endl;
      if (change_flag == 1)
      {
        //std::cout <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        //std::cout << "i_start_flag:  " << i_start_flag << std::endl;
        for (auto i = i_start_flag; i < WGD->nx-1; i++)
        {
          icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          intersect_1[icell_cent] = 0;
        }
      }

      change_flag = 0;

      for (auto i = WGD->nx-3; i >= 0; i--)
      {

        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);

        if (WGD->icellflag[icell_cent+1] == 6 && WGD->icellflag[icell_cent] != 6 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
        {
          change_flag = 1;
          i_start_flag = i;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          //std::cout << "i_start_flag:  " << i_start_flag << std::endl;
        }

        if ((change_flag == 1 && WGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && WGD->icellflag[icell_cent] == 1))
        {
          change_flag = 0;
        }
        intersect_1opp[icell_cent] = change_flag;
      }

      if (change_flag == 1)
      {
        //std::cout <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        for (auto i = WGD->nx-2; i >= i_start_flag; i--)
        {
          icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
          intersect_1opp[icell_cent] = 0;
        }
      }
      change_flag = 0;
    }
  }
  //std::cout << "WGD->icellflag:  " << WGD->icellflag[90+75*(WGD->nx-1)+1*(WGD->nx-1)*(WGD->ny-1)] << std::endl;
  //std::cout << "WGD->icellflag-1:  " << WGD->icellflag[90+74*(WGD->nx-1)+1*(WGD->nx-1)*(WGD->ny-1)] << std::endl;
  change_flag = 0;
  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto i = 0; i < WGD->nx-1; i++)
    {
      for (auto j = 1; j < WGD->ny-1; j++)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (WGD->icellflag[icell_cent-(WGD->nx-1)] == 6 && WGD->icellflag[icell_cent] != 6 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
        {
          change_flag = 1;
          j_start_flag = j;
          //std::cout << "j_start_flag:  " << j_start_flag << std::endl;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        }
        //std::cout << "j_start_flag:  " << j_start_flag << std::endl;
        if ((change_flag == 1 && WGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && WGD->icellflag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 0;
        }
        intersect_2[icell_cent] = change_flag;
      }
      if (change_flag == 1)
      {
        //std::cout <<"i:  "<< i << "\t\t" << "k:  " << k <<std::endl;
        for (auto j = j_start_flag; j < WGD->ny-1; j++)
        {
          icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
          intersect_2[icell_cent] = 0;
        }
      }
      change_flag = 0;
      for (auto j = WGD->ny-3; j >= 0; j--)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (WGD->icellflag[icell_cent+(WGD->nx-1)] == 6 && WGD->icellflag[icell_cent] != 6 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 1;
          j_start_flag = j;
        }
        if ((change_flag == 1 && WGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && WGD->icellflag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 0;
        }
        intersect_2opp[icell_cent] = change_flag;
      }

      if (change_flag == 1)
      {
        for (auto j = WGD->ny-2; j >= j_start_flag; j--)
        {
          icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
          intersect_2opp[icell_cent] = 0;
        }
      }
      change_flag = 0;
    }
  }

  //std::cout << "intersect_2:  " << intersect_2[90+75*(WGD->nx-1)+1*(WGD->nx-1)*(WGD->ny-1)] << std::endl;


  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto j = 0; j < WGD->ny-1; j++)
    {
      for (auto i = 0; i < WGD->nx-1; i++)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if ((intersect_1[icell_cent] == 1 || intersect_1opp[icell_cent] == 1) && (intersect_2[icell_cent] == 1 || intersect_2opp[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          intersect[icell_cent] = 1;
        }
      }
    }
  }

  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto j = 1; j < WGD->ny-1; j++)
    {
      NS_flag = 0;
      for (auto i = 1; i < WGD->nx-1; i++)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (intersect[icell_cent] == 1 && WGD->icellflag[icell_cent-1] == 6)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          NS_flag = 1;
        }
        if (intersect[icell_cent] != 1 && NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          NS_flag = 0;
        }
        if (NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          E_W_flag[icell_cent] = 1;
        }
      }
      NS_flag = 0;
      for (auto i = WGD->nx-3; i >= 0; i--)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (intersect[icell_cent] == 1 && WGD->icellflag[icell_cent+1] == 6)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          NS_flag = 1;
        }
        if (intersect[icell_cent] != 1 && NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          NS_flag = 0;
        }
        if (NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          W_E_flag[icell_cent] = 1;
        }
      }
    }
  }


  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto i = 1; i < WGD->nx-1; i++)
    {
      NS_flag = 0;
      for (auto j = 1; j < WGD->ny-1; j++)
      {
        //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (intersect[icell_cent] == 1 && WGD->icellflag[icell_cent-(WGD->nx-1)] == 6)
        {
          NS_flag = 1;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        }
        if (intersect[icell_cent] != 1 && NS_flag == 1)
        {
          NS_flag = 0;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        }
        //std::cout << "NS_flag:  " << NS_flag << std::endl;
        if (NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          S_N_flag[icell_cent] = 1;
        }
      }
      NS_flag = 0;
      for (auto j = WGD->ny-3; j >= 0; j--)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if (intersect[icell_cent] == 1 && WGD->icellflag[icell_cent+(WGD->nx-1)] == 6)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          NS_flag = 1;
        }
        if (intersect[icell_cent] != 1 && NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          NS_flag = 0;
        }
        if (NS_flag == 1)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          N_S_flag[icell_cent] = 1;
        }
      }
    }
  }

  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto j = 0; j < WGD->ny-1; j++)
    {
      for (auto i = 0; i < WGD->nx-1; i++)
      {
        icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        if ((E_W_flag[icell_cent] == 1 || S_N_flag[icell_cent] == 1) && (N_S_flag[icell_cent] == 1 || W_E_flag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          WGD->icellflag[icell_cent] = 12;
        }
      }
    }
  }

}*/


/**
 *
 * This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
 * This function reads in building features like nodes, building height and base height and uses
 * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
 * cells qualified on top of buildings and applies the approperiate parameterization to them.
 * More information:
 *
 */
/*void PolyBuilding::poisson (const WINDSInputData* WID, WINDSGeneralData* WGD)
{

  for (auto iter = 0; iter < 10; iter++)
  {

    for (auto k = 1; k < WGD->nz-1; k++)
    {
      for (auto j = 1; j < WGD->ny-1; j++)
      {
        for (auto i = 1; i < WGD->nx-1; i++)
        {
          icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
          icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;
          if (WGD->icellflag[icell_cent] == 12 && WGD->icellflag[icell_cent-1] == 12)
          {
            WGD->u0[icell_face] = (1 / ( WGD->e[icell_cent] + WGD->f[icell_cent] + WGD->g[icell_cent] +
                                            WGD->h[icell_cent] + WGD->m[icell_cent] + WGD->n[icell_cent])) *
                ( WGD->e[icell_cent] * WGD->u0[icell_face+1]        + WGD->f[icell_cent] * WGD->u0[icell_face-1] +
                  WGD->g[icell_cent] * WGD->u0[icell_face + WGD->nx] + WGD->h[icell_cent] * WGD->u0[icell_face-WGD->nx] +
                  WGD->m[icell_cent] * WGD->u0[icell_face + WGD->nx*WGD->ny] +
                  WGD->n[icell_cent] * WGD->u0[icell_face - WGD->nx*WGD->ny] );
                  //std::cout << "u0:  " << WGD->u0[icell_face] << std::endl;
          }

          if (WGD->icellflag[icell_cent] == 12 && WGD->icellflag[icell_cent-(WGD->nx-1)] == 12)
          {
            WGD->v0[icell_face] = (1 / ( WGD->e[icell_cent] + WGD->f[icell_cent] + WGD->g[icell_cent] +
                                            WGD->h[icell_cent] + WGD->m[icell_cent] + WGD->n[icell_cent])) *
                ( WGD->e[icell_cent] * WGD->v0[icell_face+1]        + WGD->f[icell_cent] * WGD->v0[icell_face-1] +
                  WGD->g[icell_cent] * WGD->v0[icell_face + WGD->nx] + WGD->h[icell_cent] * WGD->v0[icell_face-WGD->nx] +
                  WGD->m[icell_cent] * WGD->v0[icell_face + WGD->nx*WGD->ny] +
                  WGD->n[icell_cent] * WGD->v0[icell_face - WGD->nx*WGD->ny] );
                  //std::cout << "v0:  " << WGD->v0[icell_face] << std::endl;
          }

          if (WGD->icellflag[icell_cent] == 12 && WGD->icellflag[icell_cent-(WGD->nx-1)*(WGD->ny-1)] == 12)
          {
            WGD->w0[icell_face] = (1 / ( WGD->e[icell_cent] + WGD->f[icell_cent] + WGD->g[icell_cent] +
                                            WGD->h[icell_cent] + WGD->m[icell_cent] + WGD->n[icell_cent])) *
                ( WGD->e[icell_cent] * WGD->w0[icell_face+1]        + WGD->f[icell_cent] * WGD->w0[icell_face-1] +
                  WGD->g[icell_cent] * WGD->w0[icell_face + WGD->nx] + WGD->h[icell_cent] * WGD->w0[icell_face-WGD->nx] +
                  WGD->m[icell_cent] * WGD->w0[icell_face + WGD->nx*WGD->ny] +
                  WGD->n[icell_cent] * WGD->w0[icell_face - WGD->nx*WGD->ny] );
          }

        }
      }
    }

  }
}*/
