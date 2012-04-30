/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Put boundary matrices, their types and encoding/decoding in one place.
*/

#ifndef BOUNDARY_MATRICES
#define BOUNDARY_MATRICES 1

#include "celltypes.h"

// CUDA head file, right?
#include <vector_types.h>
#include <iostream>

namespace QUIC
{

	// Naming collisions? Can this be put somewhere not to have that happen?	
	// The application of any encoding function is orthogonal to any other.
	enum BOUNDARYMASKS
	{
		// Domain edge
    M_DOMAIN_M     = 7168, // The following three...
    M_DOMAIN_SLC_M = 4096,
    M_DOMAIN_ROW_M = 2048,
    M_DOMAIN_COL_M = 1024,

    // Wall bits
		M_E_M = 512, 
		M_F_M = 256, 
		M_G_M = 128, 
		M_H_M =  64,
		M_M_M =  32, 
		M_N_M =  16,
		M_O_M =   8, 
		M_P_M =   4, 
		M_Q_M =   2,
		
    // Red or Black pass    
    M_REDPASS_M    =  1,
	};

	typedef struct boundaryMatrices
	{
		int* cmprssd;  // e,f,g,h,m,n,o,p and q in one matrix.
		float* denoms; // omegarelax and 2(o + Ap + Bq)
		
		uint3 dim;
	} boundaryMatrices;


  __host__ __device__
  static inline bool isDomainBoundary
  (
    unsigned const slcI, unsigned const rowI, unsigned const colI, 
    uint3 const dim, bool& slcBndry, bool& rowBndry, bool& colBndry
  )
  {
    slcBndry = (slcI == 0 || slcI == dim.z - 1);
		rowBndry = (rowI == 0 || rowI == dim.y - 1);
		colBndry = (colI == 0 || colI == dim.x - 1);
		
		return slcBndry || rowBndry || colBndry;
  }
  
	
	// Used to encode boundaries for a specific cell into an integer
	__host__ __device__ 
	static inline void encodeBoundary
	(
		int& cmprssd,
		float const e, float const f, 
		float const g, float const h,
		float const m, float const n,
		float const o, float const p, float const& q
	)
	{
		if (e == 1.f) {cmprssd += M_E_M;}
		if (f == 1.f) {cmprssd += M_F_M;}
		if (g == 1.f) {cmprssd += M_G_M;}
		if (h == 1.f) {cmprssd += M_H_M;}
		if (m == 1.f) {cmprssd += M_M_M;}
		if (n == 1.f) {cmprssd += M_N_M;}
		if (o == 1.f) {cmprssd += M_O_M;}
		if (p == 1.f) {cmprssd += M_P_M;}
		if (q == 1.f) {cmprssd += M_Q_M;}
	}

	// Used to decode boundaries from an integer in a specific cell to float values.
	__host__ __device__	
	static inline void decodeBoundary
	(
    int bndry, 
    float& e, float& f, 
    float& g, float& h, 
    float& m, float& n, 
    float& o, float& p, float& q
	)
	{
	  // Shift off the pass mask.
	  bndry >>= 1;
	
    q = .5f*(bndry & 1) + .5f; bndry >>= 1;
    p = .5f*(bndry & 1) + .5f; bndry >>= 1;
    o = .5f*(bndry & 1) + .5f; bndry >>= 1;

    n = 1.f*(bndry & 1); bndry >>= 1;
    m = 1.f*(bndry & 1); bndry >>= 1;
    h = 1.f*(bndry & 1); bndry >>= 1;
    g = 1.f*(bndry & 1); bndry >>= 1;
    f = 1.f*(bndry & 1); bndry >>= 1;
    e = 1.f*(bndry & 1);
	}
	
	__host__ __device__
	static inline void encodeDomainBoundaryMask
	(
		int& cmprssd, 
		bool const slc_mask, 
		bool const row_mask,
		bool const col_mask
	)
	{		
    if (slc_mask) {cmprssd += M_DOMAIN_SLC_M;} 
		if (row_mask) {cmprssd += M_DOMAIN_ROW_M;}
		if (col_mask) {cmprssd += M_DOMAIN_COL_M;}
	}

	__host__ __device__ 
	static inline bool decodeDomainMask(int const cmprssd)
	{
		return cmprssd & M_DOMAIN_M;
	}

	__host__ __device__
	static inline bool decodeDomainSliceMask(int const cmprssd)
	{
		return cmprssd & M_DOMAIN_SLC_M;
	}

	__host__ __device__
	static inline bool decodeDomainRowMask(int const cmprssd)
	{
		return cmprssd & M_DOMAIN_ROW_M;
	}

	__host__ __device__
	static inline bool decodeDomainColMask(int const cmprssd)
	{
		return cmprssd & M_DOMAIN_COL_M;
	}

	__host__ __device__
	static inline void encodePassMask(int& cmprssd, bool const pass_mask)
	{
		cmprssd |= (pass_mask) ? M_REDPASS_M : cmprssd ;
	}

	__host__ __device__
	static inline bool decodePassMask(int const cmprssd)
	{
		return cmprssd & M_REDPASS_M;
	}
	
	// Used to determine a boundary cell from a type cell.
  // Cmprssd is the integer that will store the encoded boundary information.
  // typs is a Celltype pointer to the CURRENT type cell for encoding.
  // NOTE: typs is accessed above, below, front, behind, right and left of the 
  //       current cell being pointed at. --> Maybe not the best design...
	__host__ __device__ 
	static inline void determineBoundaryCell
	(
	  int& cmprssd, celltypes const& typs, int const ndx
	)
  {
		int row = typs.dim.x;
		int slc = typs.dim.x*typs.dim.y;
	
		int wslcI = ndx % slc;
		
		int rowI  = wslcI / row;
		int colI  = wslcI % row;
		int slcI  = ndx / slc;
		
		bool slcBndry, rowBndry, colBndry; // z, y, x...

    // Start with a clean slate.
		cmprssd = 0;

    // All interior domain cells start as fluid cells.
    float e, f, g, h, m, n, o, p, q;
		e = f = g = h = m = n = o = p = q = 1.f;
    // TODO Should SOLID cells be the same as fluid? Or all zeroed?

		if(QUIC::isDomainBoundary(slcI, rowI, colI, typs.dim, slcBndry, rowBndry, colBndry))
		{
			e = f = g = h = m = n = p = q = 0.f;
			o = 1.f;
			QUIC::encodeDomainBoundaryMask(cmprssd, slcBndry, rowBndry, colBndry);
		}
		else if(typs.c[ndx] != SOLID)
		{
			// Unless they have a solid cell as a neighbor in any direction.
			// OR are a solid or domain boundary cell.
			bool w_lft = typs.c[ndx -   1] == SOLID; // wall to left
			bool w_rgt = typs.c[ndx +   1] == SOLID; // wall to right
			bool w_bck = typs.c[ndx - row] == SOLID; // wall to back or behind
			bool w_fwd = typs.c[ndx + row] == SOLID; // wall to forward or front
			bool w_dwn = typs.c[ndx - slc] == SOLID; // wall to down or below
			bool w_upp = typs.c[ndx + slc] == SOLID; // wall to up or above			
	
			e = (w_rgt) ? 0.f : 1.f ;
			f = (w_lft) ? 0.f : 1.f ;
			g = (w_fwd) ? 0.f : 1.f ;
			h = (w_bck) ? 0.f : 1.f ;
			m = (w_upp) ? 0.f : 1.f ;
			n = (w_dwn) ? 0.f : 1.f ;
		
			o = (w_lft || w_rgt) ? .5f : 1.f ;
			p = (w_bck || w_fwd) ? .5f : 1.f ;
			q = (w_dwn || w_upp) ? .5f : 1.f ;
		}

		// encode / compress the boundaries.
		QUIC::encodeBoundary(cmprssd, e, f, g, h, m, n, o, p, q);
		QUIC::encodePassMask(cmprssd, (colI + rowI + slcI) & 1);			
	}
}

#endif

