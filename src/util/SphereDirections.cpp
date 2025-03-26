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
 * @file SphereDirections.cpp
 * @brief :document this:
 */

#include "SphereDirections.h"

#if 0
SphereDirections::SphereDirections(){
   //default cardinal directions for now
   nextList[0] = *(new Vector3(1,0,0));   //front
   nextList[1] = *(new Vector3(-1,0,0));  //back
   nextList[2] = *(new Vector3(0,1,0));   //left
   nextList[3] = *(new Vector3(0,-1,0));  //right
   nextList[4] = *(new Vector3(0,0,1));   //top
   nextList[5] = *(new Vector3(0,0,-1));  //bottom
   vecCount = 0;
   numDirVec = 6;
}

SphereDirections::SphereDirections(int numDirVec){
   //default cardinal directions for now
   nextList[0] = Vector3(1,0,0);   //front
   nextList[1] = Vector3(-1,0,0);  //back
   nextList[2] = Vector3(0,1,0);   //left
   nextList[3] = Vector3(0,-1,0);  //right
   nextList[4] = Vector3(0,0,1);   //top
   nextList[5] = Vector3(0,0,-1);  //bottom
   vecCount = 0;
   this->numDirVec = numDirVec;
}


SphereDirections::SphereDirections(int numDirVec, float lowerThetaBound, float upperThetaBound, float lowerPhiBound, float upperPhiBound){

   vecCount = 0;
   this->numDirVec = numDirVec;
   this->lowerThetaBound = lowerThetaBound;
   this->upperThetaBound = upperThetaBound;
   this->lowerPhiBound = lowerPhiBound;
   this->upperPhiBound = upperPhiBound;

   //just to for sure check the bottom direction
   nextList[0] = *(new Vector3(0,0,-1));
}

Vector3 SphereDirections::getNextDirCardinal(){
   //temp for 6 cardinal directions
   Vector3* next = NULL;
   if(vecCount < numDirVec){
      next = &nextList[vecCount];
      vecCount++;
   }
   std::cout<<"Next Direction: "<<(*next)[0]<<", "<<(*next)[1]<<", "<<(*next)[2]<<std::endl;
   return *next;
}



Vector3 SphereDirections::getNextDir2(){
   Vector3 nextDir;

   //testing Mitchell's Best Candidate Version
   if(vecCount < numDirVec){

      if(vecCount == 0){ //generate the first reference point for the MBC
         prevDir = nextList[5]; //set to bottom dir
      }


      if(numDirVec > 6 && vecCount < 6){
         nextDir = nextList[vecCount];
      }else{
         int N = 5; //number of sample points to create
         Vector3 samplePtList[N];
         for(int i = 0; i < N; i++){
            samplePtList[i] = genRandSpherePt();
         }

         nextDir = furthestPoint(samplePtList, prevDir, N);
         prevDir = nextDir;
      }
   }else{
      std::cout<<"You are trying to get more directional vectors than indicated."<<std::endl;
   }

   return nextDir;
}

#endif
