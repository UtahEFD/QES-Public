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


SphereDirections::SphereDirections(int numDir, Vector3<float> bound_vect_x,
                                   Vector3<float> bound_vec_y){

   vecCount = 0;

   //TODO: Set bounds to phi and theta limits instead
   //Generate random directional vectors by generating random points
   //on the surface of a sphere radius 1
   //Since the radius is 1, the maginitude of all directional vectors
   //should be 1, meaning that each point generated should be a unit
   //vector
   //The number of generated rays should be numDir, which should be
   //large enough to generate a fairly decent spread of rays to test

}


Vector3<float> SphereDirections::getNextDir(){
   //temp for 6 cardinal directions
   Vector3<float> next = nextList[vecCount];
   vecCount++;

   return next;

}
