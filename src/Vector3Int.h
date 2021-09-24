/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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


#pragma once

#include <iostream>
#include "util/ParseInterface.h"


class Vector3Int : public ParseInterface
{
protected:
  std::vector<int> values;


public:
  Vector3Int()
  {

    values.resize(3);
    values[0] = values[1] = values[2] = 0;
  }

  template<typename X>
  Vector3Int(const X a, const X b, const X c)
  {
    /*values.clear();
    values.push_back(a);
    values.push_back(b);
    values.push_back(c);*/
    values.resize(3);

    values[0] = a;
    values[1] = b;
    values[2] = c;
  }

  virtual void parseValues()
  {
    values.clear();
    parseTaglessValues(values);
  }

  ~Vector3Int() {}

  int operator[](const int i) const
  {
    return values[i];
  }

  int &operator[](const int i)
  {
    return values[i];
  }
};
