#include SphereDirections.h

SphereDirections::SphereDirections(){
   //default cardinal directions for now
   nextList[0] = new Vector3(1,0,0);
   nextList[1] = new Vector3(-1,0,0);
   nextList[2] = new Vector3(0,1,0);
   nextList[3] = new Vector3(0,-1,0);
   nextList[4] = new Vector3(0,0,1);
   nextList[5] = new Vector3(0,0,-1);
   vecCount = 0;
}


SphereDirections::SphereDirections(int numDirVec, float lowerThetaBound, float upperThetaBound, float lowerPhiBound, float upperPhiBound){

   vecCount = 0;
   this.numDirVec = numDirVec;
   this.lowerThetaBound = lowerThetaBound;
   this.upperThetaBound = upperThetaBound;
   this.lowerPhiBound = lowerPhiBound;
   this.upperPhiBound = upperPhiBound;

}


Vector3<float> SphereDirections::getNextDirCardinal(){
   //temp for 6 cardinal directions
   Vector3<float> next = nextList[vecCount];
   vecCount++;
   return next;
}


Vector2<float> SphereDirections::getNextDir(){
   //will only work for c++11 and up since I used the
   //mersenne_twiser_engine to generate uniform rand num
   //Might need to make sure it is actually generating a uniform rand
   //float num

   std::random_device rd;
   std::mt19937 e2(rd());
   std::uniform_real_distribution<float> theta(lowerThetaBound, upperThetaBound);
   std::uniform_real_distribution<float> phi(lowerPhiBound, upperThetaBound);

   float dx = std::cos(theta(e2))*std::cos(phi(e2));
   float dy = std::sin(phi(e2));
   float dz = std::cos(theta(e2))*std::sin(phi(e2));

   return new Vector3<float> (dx,dy,dz);
}
