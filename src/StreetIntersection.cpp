#include "PolyBuilding.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"




/**
*
* This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
* This function reads in building features like nodes, building height and base height and uses
* features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
* cells qualified on top of buildings and applies the approperiate parameterization to them.
* More information:
*
*/
/*void PolyBuilding::streetIntersection (const URBInputData* UID, URBGeneralData* UGD)
{

  int NS_flag;
  int change_flag = 0;
  int i_start_flag, j_start_flag;
  std::vector<int> intersect, intersect_1, intersect_2;
  std::vector<int> intersect_1opp, intersect_2opp;
  std::vector<int> E_W_flag, W_E_flag, N_S_flag, S_N_flag;

  intersect.resize (UGD->numcell_cent, 0);
  intersect_1.resize (UGD->numcell_cent, 0);
  intersect_2.resize (UGD->numcell_cent, 0);
  intersect_1opp.resize (UGD->numcell_cent, 0);
  intersect_2opp.resize (UGD->numcell_cent, 0);

  E_W_flag.resize (UGD->numcell_cent, 0);
  W_E_flag.resize (UGD->numcell_cent, 0);
  N_S_flag.resize (UGD->numcell_cent, 0);
  S_N_flag.resize (UGD->numcell_cent, 0);

  UGD->wall->setVelocityZero (UGD);

  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto j = 0; j < UGD->ny-1; j++)
    {
      for (auto i = 1; i < UGD->nx-1; i++)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (UGD->icellflag[icell_cent-1] == 6 && UGD->icellflag[icell_cent] != 6 && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
        {
          change_flag = 1;
          i_start_flag = i;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        }
        if ((change_flag == 1 && UGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && UGD->icellflag[icell_cent] == 1))
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
        for (auto i = i_start_flag; i < UGD->nx-1; i++)
        {
          icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          intersect_1[icell_cent] = 0;
        }
      }

      change_flag = 0;

      for (auto i = UGD->nx-3; i >= 0; i--)
      {

        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);

        if (UGD->icellflag[icell_cent+1] == 6 && UGD->icellflag[icell_cent] != 6 && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
        {
          change_flag = 1;
          i_start_flag = i;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          //std::cout << "i_start_flag:  " << i_start_flag << std::endl;
        }

        if ((change_flag == 1 && UGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && UGD->icellflag[icell_cent] == 1))
        {
          change_flag = 0;
        }
        intersect_1opp[icell_cent] = change_flag;
      }

      if (change_flag == 1)
      {
        //std::cout <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        for (auto i = UGD->nx-2; i >= i_start_flag; i--)
        {
          icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
          intersect_1opp[icell_cent] = 0;
        }
      }
      change_flag = 0;
    }
  }
  //std::cout << "UGD->icellflag:  " << UGD->icellflag[90+75*(UGD->nx-1)+1*(UGD->nx-1)*(UGD->ny-1)] << std::endl;
  //std::cout << "UGD->icellflag-1:  " << UGD->icellflag[90+74*(UGD->nx-1)+1*(UGD->nx-1)*(UGD->ny-1)] << std::endl;
  change_flag = 0;
  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto i = 0; i < UGD->nx-1; i++)
    {
      for (auto j = 1; j < UGD->ny-1; j++)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (UGD->icellflag[icell_cent-(UGD->nx-1)] == 6 && UGD->icellflag[icell_cent] != 6 && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
        {
          change_flag = 1;
          j_start_flag = j;
          //std::cout << "j_start_flag:  " << j_start_flag << std::endl;
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        }
        //std::cout << "j_start_flag:  " << j_start_flag << std::endl;
        if ((change_flag == 1 && UGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && UGD->icellflag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 0;
        }
        intersect_2[icell_cent] = change_flag;
      }
      if (change_flag == 1)
      {
        //std::cout <<"i:  "<< i << "\t\t" << "k:  " << k <<std::endl;
        for (auto j = j_start_flag; j < UGD->ny-1; j++)
        {
          icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
          intersect_2[icell_cent] = 0;
        }
      }
      change_flag = 0;
      for (auto j = UGD->ny-3; j >= 0; j--)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (UGD->icellflag[icell_cent+(UGD->nx-1)] == 6 && UGD->icellflag[icell_cent] != 6 && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 1;
          j_start_flag = j;
        }
        if ((change_flag == 1 && UGD->icellflag[icell_cent] == 6) || (change_flag == 1 && (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2))
            || (change_flag == 1 && UGD->icellflag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          change_flag = 0;
        }
        intersect_2opp[icell_cent] = change_flag;
      }

      if (change_flag == 1)
      {
        for (auto j = UGD->ny-2; j >= j_start_flag; j--)
        {
          icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
          intersect_2opp[icell_cent] = 0;
        }
      }
      change_flag = 0;
    }
  }

  //std::cout << "intersect_2:  " << intersect_2[90+75*(UGD->nx-1)+1*(UGD->nx-1)*(UGD->ny-1)] << std::endl;


  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto j = 0; j < UGD->ny-1; j++)
    {
      for (auto i = 0; i < UGD->nx-1; i++)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if ((intersect_1[icell_cent] == 1 || intersect_1opp[icell_cent] == 1) && (intersect_2[icell_cent] == 1 || intersect_2opp[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          intersect[icell_cent] = 1;
        }
      }
    }
  }

  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto j = 1; j < UGD->ny-1; j++)
    {
      NS_flag = 0;
      for (auto i = 1; i < UGD->nx-1; i++)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (intersect[icell_cent] == 1 && UGD->icellflag[icell_cent-1] == 6)
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
      for (auto i = UGD->nx-3; i >= 0; i--)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (intersect[icell_cent] == 1 && UGD->icellflag[icell_cent+1] == 6)
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


  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto i = 1; i < UGD->nx-1; i++)
    {
      NS_flag = 0;
      for (auto j = 1; j < UGD->ny-1; j++)
      {
        //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (intersect[icell_cent] == 1 && UGD->icellflag[icell_cent-(UGD->nx-1)] == 6)
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
      for (auto j = UGD->ny-3; j >= 0; j--)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if (intersect[icell_cent] == 1 && UGD->icellflag[icell_cent+(UGD->nx-1)] == 6)
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

  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto j = 0; j < UGD->ny-1; j++)
    {
      for (auto i = 0; i < UGD->nx-1; i++)
      {
        icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        if ((E_W_flag[icell_cent] == 1 || S_N_flag[icell_cent] == 1) && (N_S_flag[icell_cent] == 1 || W_E_flag[icell_cent] == 1))
        {
          //std::cout << "i:  "<< i << "\t\t" <<"j:  "<< j << "\t\t" << "k:  " << k <<std::endl;
          UGD->icellflag[icell_cent] = 12;
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
/*void PolyBuilding::poisson (const URBInputData* UID, URBGeneralData* UGD)
{

  for (auto iter = 0; iter < 10; iter++)
  {

    for (auto k = 1; k < UGD->nz-1; k++)
    {
      for (auto j = 1; j < UGD->ny-1; j++)
      {
        for (auto i = 1; i < UGD->nx-1; i++)
        {
          icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
          icell_face = i + j*UGD->nx + k*UGD->nx*UGD->ny;
          if (UGD->icellflag[icell_cent] == 12 && UGD->icellflag[icell_cent-1] == 12)
          {
            UGD->u0[icell_face] = (1 / ( UGD->e[icell_cent] + UGD->f[icell_cent] + UGD->g[icell_cent] +
                                            UGD->h[icell_cent] + UGD->m[icell_cent] + UGD->n[icell_cent])) *
                ( UGD->e[icell_cent] * UGD->u0[icell_face+1]        + UGD->f[icell_cent] * UGD->u0[icell_face-1] +
                  UGD->g[icell_cent] * UGD->u0[icell_face + UGD->nx] + UGD->h[icell_cent] * UGD->u0[icell_face-UGD->nx] +
                  UGD->m[icell_cent] * UGD->u0[icell_face + UGD->nx*UGD->ny] +
                  UGD->n[icell_cent] * UGD->u0[icell_face - UGD->nx*UGD->ny] );
                  //std::cout << "u0:  " << UGD->u0[icell_face] << std::endl;
          }

          if (UGD->icellflag[icell_cent] == 12 && UGD->icellflag[icell_cent-(UGD->nx-1)] == 12)
          {
            UGD->v0[icell_face] = (1 / ( UGD->e[icell_cent] + UGD->f[icell_cent] + UGD->g[icell_cent] +
                                            UGD->h[icell_cent] + UGD->m[icell_cent] + UGD->n[icell_cent])) *
                ( UGD->e[icell_cent] * UGD->v0[icell_face+1]        + UGD->f[icell_cent] * UGD->v0[icell_face-1] +
                  UGD->g[icell_cent] * UGD->v0[icell_face + UGD->nx] + UGD->h[icell_cent] * UGD->v0[icell_face-UGD->nx] +
                  UGD->m[icell_cent] * UGD->v0[icell_face + UGD->nx*UGD->ny] +
                  UGD->n[icell_cent] * UGD->v0[icell_face - UGD->nx*UGD->ny] );
                  //std::cout << "v0:  " << UGD->v0[icell_face] << std::endl;
          }

          if (UGD->icellflag[icell_cent] == 12 && UGD->icellflag[icell_cent-(UGD->nx-1)*(UGD->ny-1)] == 12)
          {
            UGD->w0[icell_face] = (1 / ( UGD->e[icell_cent] + UGD->f[icell_cent] + UGD->g[icell_cent] +
                                            UGD->h[icell_cent] + UGD->m[icell_cent] + UGD->n[icell_cent])) *
                ( UGD->e[icell_cent] * UGD->w0[icell_face+1]        + UGD->f[icell_cent] * UGD->w0[icell_face-1] +
                  UGD->g[icell_cent] * UGD->w0[icell_face + UGD->nx] + UGD->h[icell_cent] * UGD->w0[icell_face-UGD->nx] +
                  UGD->m[icell_cent] * UGD->w0[icell_face + UGD->nx*UGD->ny] +
                  UGD->n[icell_cent] * UGD->w0[icell_face - UGD->nx*UGD->ny] );
          }

        }
      }
    }

  }
}*/
