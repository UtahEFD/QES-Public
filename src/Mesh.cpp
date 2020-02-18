#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

void Mesh::calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLength){

   int cellNum =0;

   for(int k = 0; k< dimZ - 1; k++) {
      for(int j = 0; j < dimY - 1; j++){
         for(int i = 0; i < dimX -1; i++){

            //calculate icell index
             int icell_idx = i + j*(dimX-1) + k*(dimY-1) * (dimX-1);

             if(icellflag[icell_idx] == 1){

               SphereDirections sd(18, -1,1, 0,2*M_PI);
               
               float maxLength = std::numeric_limits<float>::infinity();

               //ray's origin = cell's center
               Ray ray((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz);

               HitRecord hit;
               float t1 = -1;
               float t0 = 0;

               //for all possible directions, determine the distance

               for(int m = 0; m < sd.getNumDirVec(); m++){

                  // ray.setDir(sd.getNextDirCardinal());
                  ray.setDir(sd.getNextDir());

                  bool isHit = tris->rayHit(ray, t0, t1, hit);

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
