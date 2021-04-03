#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return triangleBVH->heightToTri(x,y);
}

void Mesh::calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLength){

   int cellNum =0;

#pragma acc parallel loop independent
   for(int k = 0; k< dimZ - 1; k++) {
      for(int j = 0; j < dimY - 1; j++){
         for(int i = 0; i < dimX -1; i++){

            //calculate icell index
             int icell_idx = i + j*(dimX-1) + k*(dimY-1) * (dimX-1);

             if(icellflag[icell_idx] == 1){

               SphereDirections sd(512, -1,1, 0,2*M_PI);
               
               float maxLength = std::numeric_limits<float>::infinity();

               //ray's origin = cell's center
               Ray ray((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz);

               HitRecord hit;
//               float t1 = -1;
//               float t0 = 0;

               //for all possible directions, determine the distance

               for(int m = 0; m < sd.getNumDirVec(); m++){

                  // ray.setDir(sd.getNextDirCardinal());
                  ray.setDir(sd.getNextDir());

                  bool isHit = triangleBVH->rayHit(ray, hit);

                  if(isHit){
                      // std::cout<<"Hit found."<<std::endl;

                     //compare the mixLengths
                     if(hit.hitDist < maxLength){
                        maxLength = hit.hitDist;
                        // std::cout<<"maxlength updated"<<std::endl;
                     }
                  }else{
                      // std::cout<<"Hit not found"<<std::endl;
                     //std::cout<<"Hit may not be found but hit.hitDist = "<<hit.hitDist<<std::endl;
                  }

               }

               // std::cout<<"Mixing length for this cell is "<<maxLength<<std::endl;
               //add to list of vectors
               mixingLength[icell_idx] = maxLength;
               // std::cout<<"\n\n"<<std::endl;
            }

         }
      }
   }
}


// This needs to be removed from here in next round of edits.  Just
// marking now.
void Mesh::tempOPTIXMethod(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths){
   std::cout<<"--------------Enters the tempOPTIXMethod--------------------"<<std::endl;
   //   OptixRayTrace optixRayTracer(optixTris);
   //   optixRayTracer.calculateMixingLength( mlSampleRate, dimX, dimY, dimZ, dx, dy, dz, icellflag, mixingLengths);
}
