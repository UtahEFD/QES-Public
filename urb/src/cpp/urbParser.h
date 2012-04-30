/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: C++ version for parsing input files that is somewhat more general.
*         Used with other C++ classes to get the info to module and CUDA.
* Source: Adapted from init.f90 in QUICurbv5.?
*/

#ifndef URBPARSER_H
#define URBPARSER_H

#include <iostream>
#include <iomanip>

#include "urbModule.h"

#include "quicutil/legacyFileParser.h"
#include "quicutil/standardFileParser.h"

namespace QUIC
{

	/**
	* A friend of the urbModule class, urbParser handles the major input and 
	* output operations for a given datamodule. This includes parsing the needed
	* input files, populating variables for urbModule that are needed by urbSetup
	* and urbCUDA to solve the problem described. 
	*/
	class urbParser
	{
		public:
			static void parse(urbModule* um, std::string directory);
			
			static void populate(urbModule* um, std::string const& quicProjectPath);
			
			static void dump
			(
				urbModule* um,
				std::string directory, 
				std::string prefix = "um_"
			);			

      static void generatePlumeInput(QUIC::urbModule* um, std::string directory);

			// Outputs the initial velocity data as ASCII in MATLAB friendly format.
			static void outputInitialVelocityData
			(
				urbModule* um,
				std::string directory, 
				std::string const& prefix = "um_"
			);
			
			static void printValues(QUIC::urbModule* um, std::ostream& out = std::cout);
			static void printSetupTimes(QUIC::urbModule* um, std::ostream& out = std::cout);
			static void printIterTimes(QUIC::urbModule* um, std::ostream& out = std::cout);
			static void printInfo(QUIC::urbModule* um, std::ostream& out = std::cout);

      static void outputCellTypes
			(
				const urbModule* um, std::string const& directory, 
				std::string const& prefix = "um_"
			);
			
			static void outputBoundaries
			(
				const urbModule* um, std::string const& directory,
				std::string const& prefix = "um_"
			);
			
			static void outputDivergence
			(
			  const urbModule* um, std::string const& directory,
			  std::string const& prefix = "um_"
			);
			
			static void outputDenominators
			(
			  const urbModule* um, std::string const& directory,
			  std::string const& prefix = "um_"
			);
			
			static void outputLagrangians
			(
			  const urbModule* um, std::string const& directory,
			  std::string const& prefix = "um_"
			);
			
			static void outputErrors
			(
			  const urbModule* um, std::string const& directory,
			  std::string const& prefix = "um_"
			);
			
			static void outputVelocities
			(
				const urbModule* um, std::string const& directory,
				std::string const& prefix = "um_"
			);
			
			static void outputViscocity
			(
			  const urbModule* um, std::string const& directory,
			  std::string const& prefix = "um_"
			);
			
		private:

//			static bool parse_input(urbModule* um, std::string const& directory);
			
			static void output_QU_screenout(urbModule* um, std::string const& directory);
			static void output_QP_buildout (urbModule* um, std::string const& directory);
			
			static bool output_QU_velocities (urbModule* um, std::string const& directory);
			static bool output_QU_celltypes  (urbModule* um, std::string const& directory);
			static bool output_PlumeInputFile(urbModule* um, std::string const& directory);
	};
}

#endif
