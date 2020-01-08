#include "SphereDirections.h"

SphereDirections::SphereDirections(){
   //default cardinal directions for now
   nextList[0] = *(new Vector3<float>(1,0,0));   //front
   nextList[1] = *(new Vector3<float>(-1,0,0));  //back
   nextList[2] = *(new Vector3<float>(0,1,0));   //left
   nextList[3] = *(new Vector3<float>(0,-1,0));  //right
   nextList[4] = *(new Vector3<float>(0,0,1));   //top
   nextList[5] = *(new Vector3<float>(0,0,-1));  //bottom
   vecCount = 0;
   numDirVec = 6;
}


SphereDirections::SphereDirections(int numDirVec, float lowerThetaBound, float upperThetaBound, float lowerPhiBound, float upperPhiBound){

   vecCount = 0;
   this->numDirVec = numDirVec;
   this->lowerThetaBound = lowerThetaBound;
   this->upperThetaBound = upperThetaBound;
   this->lowerPhiBound = lowerPhiBound;
   this->upperPhiBound = upperPhiBound;

   //just to for sure check the bottom direction
   nextList[0] = *(new Vector3<float>(0,0,-1));
}


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


Vector3<float> SphereDirections::getNextDir(){
   //will only work for c++11 and up since I used the
   //mersenne_twiser_engine to generate uniform rand num
   //Might need to make sure it is actually generating a uniform rand
   //float num

   Vector3<float>* next;
   if(vecCount < numDirVec - 1){
      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<float> theta(lowerThetaBound, upperThetaBound);
      std::uniform_real_distribution<float> phi(lowerPhiBound, upperPhiBound);

      float theta2 = std::asin(theta(e2));

      float dx = std::cos(theta2)*std::cos(phi(e2));
      float dy = std::sin(phi(e2));
      float dz = std::cos(theta2)*std::sin(phi(e2));

      float magnitude = std::sqrt(std::pow(dx,2)+std::pow(dy,2)+std::pow(dz,2));
      next = new Vector3<float>(dx/magnitude,dy/magnitude,dz/magnitude);
      vecCount++;
   }else if(vecCount == numDirVec -1){
      return Vector3<float>(0,0,-1);
   }else{
      next = NULL;
   }

   return *next;
}

Vector3<float> SphereDirections::getNextDir2(){
   Vector3<float> nextDir;

   //currently in progress

   return nextDir;
}

int SphereDirections::getNumDirVec(){return numDirVec;}
