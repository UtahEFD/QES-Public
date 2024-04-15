/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file QESout.cpp */

#include "QESout.h"

namespace QESout {
namespace {
  bool verbose_flag = false;
}
void splashScreen()
{
  std::cout << "###################################################################" << std::endl;
  std::cout << "#                                                                 #" << std::endl;
  std::cout << "#                        Welcome to QES                           #" << std::endl;
  std::cout << "#                                                                 #" << std::endl;
  std::cout << "###################################################################" << std::endl;
  std::cout << "Version: " << QES_VERSION_INFO << std::endl;// QES_VERSION_INFO comes from CMakeLists.txt
#ifdef HAS_CUDA
  std::cout << "\t* CUDA support available!" << std::endl;
#else
  std::cout << "* No CUDA support - CPU Only Computations!" << std::endl;
#endif

#ifdef HAS_OPTIX
  std::cout << "\t* OptiX is available!" << std::endl;
#endif

#ifdef _OPENMP
  std::cout << "* OpenMP is available!" << std::endl;
#endif
  std::cout << "###################################################################" << std::endl;
}


void error(std::string out)
{
  std::cerr << "\n===================================================================" << std::endl;
  std::cerr << "[ERROR]\t " << out << std::endl;
  std::cerr << "===================================================================" << std::endl;
  exit(EXIT_FAILURE);
}

void warning(std::string out)
{
  std::cerr << "[!! WARNING !!]\t " << out << std::endl;
}

void setVerbose()
{
  verbose_flag = true;
}

void verbose(std::string out)
{
  if (verbose_flag)
    std::cout << out << std::endl;
}
}// namespace QESout


// //////////////////////////////////////////////////
//
// IMPORTANT:  Keep the code below --Pete W
// The following code is needed to build in flags for the
// sanitizer that will allow CUDA to be run with the sanitizer
// checks in place. Without this function callback, these
// options needs to be set on the command line when the
// executable is called. For example,
// ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0 ./myExec
//
// Building the options in, as done below allows us to run the executables
// cleanly without having to type what is above.
//
#if defined(HAS_CUDA) && defined(__SANITIZE_ADDRESS__)
#ifdef __cplusplus
extern "C"
#endif
  const char *
  __asan_default_options()
{
  return "protect_shadow_gap=0:replace_intrin=0:detect_leaks=0";
}
#endif
// //////////////////////////////////////////////////
