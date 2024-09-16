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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file RandomSingleton
 * @brief This class handles the random number generation
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <chrono>

class Timer
{
public:
  // constructor
  Timer(std::string str_in) : name(str_in)
  {
    startTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tmp(0.0);
    elapsed = tmp;
  }

  ~Timer()
  {}

  void start()
  {
    startTime = std::chrono::high_resolution_clock::now();
  }

  void stop()
  {
    endTime = std::chrono::high_resolution_clock::now();
    elapsed += endTime - startTime;
  }

  void show()
  {
    std::cout << "Elapsed time for " << name.append(25 - name.length(), ' ') << elapsed.count() << " s\n";
  }

private:
  Timer()
  {}

  std::string name;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
  std::chrono::duration<double> elapsed;
};
