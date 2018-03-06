/*
 *  test_Vector3D.cpp
 *
 *  Created by Pete Willemsen on 10/6/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 * This file is part of CS5721 Computer Graphics library (cs5721Graphics).
 *
 * cs5721Graphics is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cs5721Graphics is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with cs5721Graphics.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdlib>
#include "util/Vector3D.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  Vector3D a, b(1.0, 2.0, 3.0);
  
  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;


  Vector3D c = b + b;
  std::cout << "c = " << c << ", b = " << b << std::endl;

  Vector3D d = c;
  std::cout << "d = " << d << std::endl;
  
  exit(EXIT_SUCCESS);
}
