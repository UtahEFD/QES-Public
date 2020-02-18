#include "SphereDirections.h"

#if 0

Vector3<float> SphereDirections::getNextDirCardinal(){
   //temp for 6 cardinal directions
   Vector3<float>* next = NULL;
   if(vecCount < numDirVec){
      next = &nextList[vecCount];
      vecCount++;
   }
   std::cout<<"Next Direction: "<<(*next)[0]<<", "<<(*next)[1]<<", "<<(*next)[2]<<std::endl;
   return *next;
}



Vector3<float> SphereDirections::getNextDir2(){
   Vector3<float> nextDir;

   //currently in progress

   return nextDir;
}

#endif
