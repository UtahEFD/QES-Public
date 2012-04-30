/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Timing multiple runs of CUDA iteration scheme.
*/

#ifndef URBTIMER_H
#define URBTIMER_H

#include <vector>
#include <iomanip>

#include "urbCUDA.h"

#include "quicutil/standardFileParser.h"

namespace QUIC 
{
	/**
	* Part of the QUIC namespace, the urbTimer times the module using a set break
	* down and the given data. The start of, the end of and the step of the timing
	* range is determined from an input file timing_range.txt within the running
	* directory.
	*/
	class urbTimer : public urbModule, public urbCUDA
	{
		public:
		
			urbTimer();

			virtual ~urbTimer();

			void setRange(int3 const&, int3 const&);
			int3 getStart() const;
			int3 getEnd() const;
			
			void setStep(int3 const&);
			int3 getStep() const;
			
			void clearTimings();
			float getTiming(int const&, int const&, int const&) const;
			float getTiming(float const& x, float const& y, float const& z) const;
						
			void runIterationTimings();
			void outputTimings(std::string const&) const;
			
			void printInfo() const;
			void determineParameters(std::string& inp_dir);
		
		private:
			
			int3 start;
			int3 end;
			int3 step;
			
			std::vector<float> timings;
			int3 tDim;
			
			void setTiming(int const&, int const&, int const&, float const&);
			void setTiming(float const&, float const&, float const&, float const&);
			
			void resizeTimings();
	};
}

#endif

