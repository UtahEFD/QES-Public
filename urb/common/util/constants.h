/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Common constants found throughout QUICurb.
* Remark: Is this the best place or proper?
*/

#ifndef INC_CONSTANTS_H
#define INC_CONSTANTS_H

#include <cmath>

namespace QUIC
{
	// Von Karmam Constant
	static const float VON_KARMAN = .4;
	
	// Wake Constants
	static const float FARWAKE_EXP = 1.5;
	static const float FARWAKE_FAC = 3.;
}

#endif
