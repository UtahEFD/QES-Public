#include "Mesh.h"

float Mesh::getHeight(float x, float y)
{
   return tris->heightToTri(x,y);
}

void Mesh::calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLength){

   int flag=0; //temp to just get one distribution for graph
   int cellNum =0;
   std::ofstream mixOutputFile;
   if(mixOutputFile.is_open()){
      mixOutputFile.close(); //close if previously opened
   }else{
      mixOutputFile.open("mixLengthOutput.csv");
   }
   std::ofstream cellPointsFile;
   if(cellPointsFile.is_open()){
      cellPointsFile.close();
   }else{
      cellPointsFile.open("cellPointOutput.csv");
   }

   for(int k = 0; k< dimZ - 1; k++) {
      for(int j = 0; j < dimY - 1; j++){
         for(int i = 0; i < dimX -1; i++){

            //std::cout<<"====================icell "<<k<<","<<j<<","<<i<<"================================"<<std::endl;
            //calculate icell index
            int icell_idx = i + j*(dimX-1) + k*(dimY-1) * (dimX-1);

            if(icellflag[icell_idx] == 1){
               std::cout<<"==========================================fluid icell "<<i<<","<<j<<","<<k<<" ==================================="<<std::endl;


               SphereDirections sd;
               //SphereDirections sd(1000, 0, 2*M_PI, -M_PI, M_PI);
               //SphereDirections sd(2000,-1,1, 0, 2*M_PI);
               float maxLength = std::numeric_limits<float>::infinity();

               //ray's origin = cell's center
               Ray ray((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz);
               //std::cout<<"Ray center init"<<std::endl;
               cellPointsFile<<ray.getOriginX()<<","<<ray.getOriginY()<<","<<ray.getOriginZ()<<std::endl;
               //std::cout<<"Ray origin in mesh = "<<ray.getOriginX()<<", "<<ray.getOriginY()<<", "<< y.getOriginZ()<<std::endl;
               HitRecord hit;
               float t1 = -1;
               float t0 = 0;
               //BVH* nextBVH;

               //for all possible directions, determine the distance

               std::ofstream sphereOutputFile;
               if(flag == 0){
                  sphereOutputFile.open("spherePts.csv");
               }
               for(int m = 0; m < sd.getNumDirVec(); m++){
                  std::cout<<"\n\nDirection interation #:"<<m<<std::endl;

                  ray.setDir(sd.getNextDirCardinal());
                  //ray.setDir(sd.getNextDir());

                  if(flag==0){
                     sphereOutputFile<<ray.getDirection()[0]<<","<<ray.getDirection()[1]<<","<<ray.getDirection()[2]<<std::endl;
                  }
                  std::cout<<"Direction: <"<<ray.getDirection()[0]<<", "<<ray.getDirection()[1]<<","<<ray.getDirection()[2]<<">"<<std::endl;
                  //Re-init hit?

                  bool isHit = tris->rayHit(ray, t0, t1, hit);

                  if(isHit){
                     std::cout<<"Hit found."<<std::endl;

                     //compare the mixLengths
                     if(hit.hitDist < maxLength){
                        maxLength = hit.hitDist;
                        std::cout<<"maxlength updated"<<std::endl;
                     }
                  }else{
                     std::cout<<"Hit not found"<<std::endl;
                     //std::cout<<"Hit may not be found but hit.hitDist = "<<hit.hitDist<<std::endl;
                  }

               }

               //spherical ray distribution for 1 cell
               if(flag ==0){
                  sphereOutputFile.close();
                  flag++;
               }

               //mixLength output file
               if(maxLength != std::numeric_limits<float>::infinity()){
                  mixOutputFile<<cellNum<<","<<maxLength<<std::endl;
                  cellNum++;
               }

               std::cout<<"Mixing length for this cell is "<<maxLength<<std::endl;
               //add to list of vectors
               mixingLength[icell_idx] = maxLength;
               std::cout<<"\n\n"<<std::endl;
            }

         }
      }
   }

   cellPointsFile.close();
   mixOutputFile.close();
   return mixingLengthList;

}
