#pragma once

#include <tuple>
#include <vector>

#include "util/NetCDFInput.h"

/**
 * \brief Namespace for core components within the QES framework.
 */
namespace qes {

struct QESgrid
{
  float dx;
  float dy;
  float dz;
  float dxy;

  int nx;
  int ny;
  int nz;
};

/**
 * \brief Domain data for running a simulation with QES.
 *
 *
 */
class Domain
{
private:
  Domain() = default;// cannot create empty domain

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
  Domain(const int &nx, const int &ny, const int &nz, const float &dx, const float &dy, const float &dz);

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
   * @param inputFile initialize from input file data
   */
  explicit Domain(const std::string &NetCDFFileDataName);

  /**
   *
   */
  int nx() const { return domainData.nx; }
  const int &tnx = domainData.nx;
  int ny() const { return domainData.ny; }
  int nz() const { return domainData.nz; }

  float dx() const { return domainData.dx; }
  float dy() const { return domainData.dy; }
  float dz() const { return domainData.dz; }
  float dxy() const { return domainData.dxy; }

  std::tuple<float, float, float> getDomainSize() const { return { domainData.dx, domainData.dx, domainData.dz }; }
  std::tuple<int, int, int> getDomainCellNum() const { return { domainData.nx, domainData.ny, domainData.nz }; }
  std::tuple<int, int, int> getBaseDomainCellNum() const { return { domainData.nx - 1, domainData.ny - 1, domainData.nz - 2 }; }

  qes::QESgrid getDomainInfo() const { return domainData; }
  /**
   *
   */

  /**
   *
   */

  /**
   * Returns the total number of cell centered cells in the domain.
   * This value uses the domain information that was provided in the
   * initial parameters of the simulation and does not return
   * information based on staggered grid domain.
   */
  long numCellCentered() const { return (domainData.nx - 1) * (domainData.ny - 1) * (domainData.nz - 1); }

  /**
   * Returns the total number of cell centered cells in the domain
   * in a single horizontal slice.  This value uses the domain
   * information that was provided in the initial parameters of the
   * simulation and does not return information based on staggered
   * grid domain.
   */
  long numHorizontalCellCentered() const { return (domainData.nx - 1) * (domainData.ny - 1); }

  // numcell_cent = (nx - 1) * (ny - 1) * (nz - 1);// Total number of cell-centered values in domain

  /**
   * Returns the total number of face centered cells in the domain
   * based on the internal staggered grid domain sizes.
   */
  long numFaceCentered() const { return domainData.nx * domainData.ny * domainData.nz; }

  // inlined too... d
  long face2d(const int &i, const int &j) const
  {
    return i + j * domainData.nx;
  }
  long face(const int &i, const int &j, const int &k) const
  {
    return i + j * domainData.nx + k * domainData.nx * domainData.ny;
  }
  long faceAdd(const long &curr, const int &i, const int &j, const int &k) const
  {
    return curr + i + j * domainData.nx + k * domainData.nx * domainData.ny;
  }

  long cell2d(const int &i, const int &j) const
  {
    return i + j * (domainData.nx - 1);
  }
  long cell(const int &i, const int &j, const int &k) const
  {
    return i + j * (domainData.nx - 1) + k * (domainData.nx - 1) * (domainData.ny - 1);
  }
  long cellAdd(const long &curr, const int &i, const int &j, const int &k) const
  {
    return curr + i + j * (domainData.nx - 1) + k * (domainData.nx - 1) * (domainData.ny - 1);
  }

  std::tuple<int, int, int> getCellIdx(const long &curr) const
  {
    int k = (int)(curr / ((domainData.nx - 1) * (domainData.ny - 1)));
    int j = (int)((curr - k * (domainData.nx - 1) * (domainData.ny - 1)) / (domainData.nx - 1));
    int i = curr - j * (domainData.nx - 1) - k * (domainData.nx - 1) * (domainData.ny - 1);
    return { i, j, k };
  }

  // 2d cell centered idx
  // 2d face centered idx


private:
  void defineHorizontalGrid();
  void defineVerticalGrid();
  void defineVerticalStretching(const float &);
  void defineVerticalStretching(const std::vector<float> &);

  QESgrid domainData{};

public:
  std::vector<float> z0_domain_u, z0_domain_v;
  std::vector<float> dz_array; /**< :Array contain dz values: */
  ///@{
  /** :Location of center of cell in x,y and z directions: */
  std::vector<float> x, y, z;
  ///@}
  std::vector<float> z_face; /**< :Location of the bottom face of the cell in z-direction: */
};

#if 0
// data that is tied more directly to the domain initialization
// rather specific to wind solving (e, g, g, etc...) type of data...
class DomainParameters
{
public:
  DomainParameters(DomainInfo dinfo);

    initializeElements( ... ???  );


  private:

    // Keep local copy of DomainInfo given by the construtor...
    Domain dInfo;
};
#endif

}// namespace qes
