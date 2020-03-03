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

Vector3<float> SphereDirections::furthestPoint(Vector3<float> *samplePtList, Vector3<float> existingPt, const int listSize){
   Vector3<float> farPt = existingPt; //flag, should never return this vector
   float farDist = 0;
   for(int i = 0; i< listSize; i++){
      float dist = std::sqrt(std::pow((*(samplePtList+i))[0] - existingPt[0], 2)
                             +std::pow((*(samplePtList+i))[1] - existingPt[1], 2)
                             +std::pow((*(samplePtList+i))[2] - existingPt[2], 2));

      if(dist > farDist){
         farPt = *(samplePtList+i);
      }
   }

   return farPt;
}

Vector3<float> SphereDirections::genRandSpherePt(){
   std::random_device rd;
   std::mt19937 e2(rd());
   std::uniform_real_distribution<float> theta(0, 2*M_PI);
   std::uniform_real_distribution<float> phi(0, std::nextafter(M_PI, FLT_MAX));

   return polarToCartesian(theta(e2), phi(e2));

}

Vector3<float> SphereDirections::polarToCartesian(float theta, float phi){
   //NOTE: r = 1 for a unit circle
   float x = std::cos(theta)*std::sin(phi);
   float y = std::sin(theta)*std::sin(phi);
   float z = std::cos(phi);

   //need to normalize?
   float magnitude = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

   if(magnitude < 0.9 || magnitude > 1.0){
      std::cout<<"Need to normalize in polarToCartesian function in SphereDirections"<<std::endl;
   }

   return Vector3<float>(x,y,z);
}

int SphereDirections::getNumDirVec(){return numDirVec;}
