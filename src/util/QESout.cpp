/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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
  std::cout << "##############################################################" << std::endl;
  std::cout << "#                                                            #" << std::endl;
  std::cout << "#                      Welcome to QES                        #" << std::endl;
  std::cout << "#                                                            #" << std::endl;
  std::cout << "##############################################################" << std::endl;
  std::cout << "version " << QES_VERSION << std::endl;
#ifdef HAS_OPTIX
  std::cout << "OptiX is available!" << std::endl;
#endif
}


void error(std::string out)
{
  std::cerr << "==============================================================" << std::endl;
  std::cerr << "[ERROR] " << out << std::endl;
  exit(EXIT_FAILURE);
}

void warning(std::string out)
{
  std::cerr << "[WARNING] " << out << std::endl;
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
