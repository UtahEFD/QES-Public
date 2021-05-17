#include "SphereDirections.h"

#if 0
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

SphereDirections::SphereDirections(int numDirVec){
   //default cardinal directions for now
   nextList[0] = Vector3<float>(1,0,0);   //front
   nextList[1] = Vector3<float>(-1,0,0);  //back
   nextList[2] = Vector3<float>(0,1,0);   //left
   nextList[3] = Vector3<float>(0,-1,0);  //right
   nextList[4] = Vector3<float>(0,0,1);   //top
   nextList[5] = Vector3<float>(0,0,-1);  //bottom
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



Vector3<float> SphereDirections::getNextDir2(){
   Vector3<float> nextDir;

   //testing Mitchell's Best Candidate Version
   if(vecCount < numDirVec){

      if(vecCount == 0){ //generate the first reference point for the MBC
         prevDir = nextList[5]; //set to bottom dir
      }


      if(numDirVec > 6 && vecCount < 6){
         nextDir = nextList[vecCount];
      }else{
         int N = 5; //number of sample points to create
         Vector3<float> samplePtList[N];
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

