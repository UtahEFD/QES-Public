/*This interface is intended to contain the base for all functions
 *related to acceleration structure construction and ray tracing
 *(including the mixing length calculations)
 */

class RayTraceInterface{
  public:
   /*
    *build acceleration structure
    */
   virtual void buildAS() = 0;
   virtual void calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const vector<int> &icellflag, vector<double> &mixingLengths) = 0;
};
