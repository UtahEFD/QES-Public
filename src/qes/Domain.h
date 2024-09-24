#include "QES_Data.h"

namespace qes {

  class Domain {
  public:
    Domain( int nx, int ny, int nz, float dx, float dy, float dz ) {
      domainData.nx = nx;
      domainData.ny = ny;
      domainData.nz = nz;

      domainData.dx = dx;
      domainData.dy = dy;
      domainData.dz = dz;
    }
    
    QESgrid getDomainInfo() { return domainData; }

    float minValueOfDxDy() { return 1.0f; }

    // inlined too... d
    int getFaceIdx( i, j, k ) { return the correct idx; }
    int getCellIdx( i, j, k ) { return the correct idx; }

    // get staggered grid info???
    void foo() {
      // ????
      // This is done to make reference to nx, ny and nz easier in this function
      // Vector3Int domainInfo;
      // domainInfo = *(WID->simParams->domain);
      nx = WID->simParams->domain[0];
      ny = WID->simParams->domain[1];
      nz = WID->simParams->domain[2];
      // Modify the domain size to fit the Staggered Grid used in the solver
      nx += 1;// +1 for Staggered grid
      ny += 1;// +1 for Staggered grid
      nz += 2;// +2 for staggered grid and ghost cell
    }

    // inline a lot of this so no need for local variable....
    int numCellCent() { return 12; }
    int numCellFace() { return 12; }    

  private:
    QESgrid domainData;

    // halo info???
    
  };

  // data that is tied more directly to the domain initialization
  // rather specific to wind solving (e, g, g, etc...) type of data...
  class DomainParameters {
  public:

    DomainParameters( DomainInfo dinfo );

    initializeElements( ... ???  );

    void defineHorizontalGrid();
    void defineVerticalGrid();
    void defineVerticalStretching(const float &);
    void defineVerticalStretching(const std::vector<float> &);

  private:
    // things like...
    std::vector<float> z0_domain_u, z0_domain_v;

      std::vector<float> dz_array; /**< :Array contain dz values: */
  ///@{
  /** :Location of center of cell in x,y and z directions: */
  std::vector<float> x, y, z;
  ///@}
  std::vector<float> z_face; /**< :Location of the bottom face of the cell in z-direction: */


  // Keep local copy of DomainInfo given by the construtor...
  Domain dInfo;
  };

}
