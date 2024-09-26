#pragma once

#include "QES_Data.h"

/**
 * \brief Namespace for core components within the QES framework.
 */
namespace qes {

/**
 * \brief Domain data for running a simulation with QES.
 *
 *
 */
class Domain
{
private:
  Domain() {}// cannot create empty domain

public:
  /**
   * /brief Standard domain constructor.
   *
   * The standard domain constructor takes the base level
   * description for the domain in terms of number of cell in the X,
   * Y and Z dimensions as arguments (nx, ny, nz). It also takes the
   * discretization in X, Y and Z dimensions as arguments (dx, dy,
   * dz).
   *
   * Internally, the class will manage this information as a staggered grid.
   *
   * @param nx Number of cells in the X dimension
   * @param ny Number of cells in the Y dimension
   * @param nz Number of cells in the Z dimension
   * @param dx Size of cells in the X dimension
   * @param dy Size of cells in the Y dimension
   * @param dz Size of cells in the Z dimension
   */
  Domain(int nx, int ny, int nz, float dx, float dy, float dz)
  {
    // Internally this information is converted to the staggered grid
    // representation, which adds 1 to nx and ny and 2 to nz (to
    // account for staggered grid and ghost cell).
    domainData.nx = nx + 1;
    domainData.ny = ny + 1;
    domainData.nz = nz + 2;

    domainData.dx = dx;
    domainData.dy = dy;
    domainData.dz = dz;
  }

  std::tuple<int, int, int> getBaseDomainCellNum() const { return { domainData.nx - 1, domainData.ny - 1, domainData.nz - 2 }; }
  std::tuple<float, float, float> getDomainSizee() const { return { domainData.dx domainData.dx, domainData.dz }; }

  float minDxy() const { return std::min(domainData.dx, domainData.dy); }

  /**
   * Returns the total number of cell centered cells in the domain.
   * This value uses the domain information that was provided in the
   * initial parameters of the simulation and does not return
   * information based on staggered grid domain.
   */
  long numCellCentered() const { return (domainData.nx - 1) * (domainData.ny - 1) * (domainData.dz - 2); }

  /**
   * Returns the total number of cell centered cells in the domain
   * in a single horizontal slice.  This value uses the domain
   * information that was provided in the initial parameters of the
   * simulation and does not return information based on staggered
   * grid domain.
   */
  long numHorizontalCellCentered() const { return (domainData.nx - 1) * (domainData.ny - 1); }

  numcell_cent = (nx - 1) * (ny - 1) * (nz - 1);// Total number of cell-centered values in domain

  /**
   * Returns the total number of face centered cells in the domain
   * based on the internal staggered grid domain sizes.
   */
  long numFaceCentered() const { return domainData.nx * domainData.ny * domainData.nz; }


  // inlined too... d
  int getFaceIdx(i, j, k) { return the correct idx; }
  int getCellIdx(i, j, k) { return the correct idx; }

private:
  QESgrid domainData;
};

// data that is tied more directly to the domain initialization
// rather specific to wind solving (e, g, g, etc...) type of data...
class DomainParameters
{
public:
  DomainParameters(DomainInfo dinfo);

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

}// namespace qes
