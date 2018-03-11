/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Porting Fortran to C++.
* Source: bisect.f90 from QUICurbv5.?
*/

#ifndef BISECT
#define BISECT

#include "constants.h"

inline static float bisect
(
	float const& ustar,
	float const& zo,
	float const& H,
	float const& ac,
	float const& psi_m
)
{
	float tol = zo / 100.f;
	float fnew = tol*10.f;
	
	float d1 = zo;
	float d2 = H;
	float d = (d1 + d2) / 2.f;

	float uhc = (ustar / VON_KARMAN)*(log((H - d1) / zo) + psi_m);
	float fi  = ac*uhc*VON_KARMAN / ustar -  H / (H - d1);

	int iter = 0;		
	while(iter < 200 && fabs(fnew) > tol)
	{
		d = (d1 + d2) / 2.f;

		uhc = (ustar / VON_KARMAN)*(log((H - d) / zo) + psi_m);
		fnew = ac*uhc*VON_KARMAN / ustar -  H/(H - d);

		if(fnew*fi > 0.f)
		{
			d1 = d;
		}
		else if(fnew*fi < 0.f)
		{
			d2 = d;
		}
		
		iter++;
	}
	
	return d;
}

#endif
