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

/** @file ParseVector.h */

#pragma once

#include <type_traits>
#include <iostream>
#include "util/ParseInterface.h"

#define FLOATS_ARE_EQUAL(x, y) (((x) - (y)) < 0.000001 && ((x) - (y)) > -0.000001)

/**
 * @class ParseVector
 * @brief Template class that holds 3 values.
 *
 * Values can be accessed as if this was an array.
 */
template<class T>
class ParseVector : public ParseInterface
{
protected:
  std::vector<T> values;

public:
  ParseVector()
  {
    values.clear();
  }


  virtual void parseValues()
  {
    values.clear();
    parseTaglessValues<T>(values);
  }

  size_t size()
  {
    return values.size();
  }

  /**
   * Accesses the value at position i.
   *
   * @param i the index of the value to return
   * @return a reference to the value stored at i
   */
  T &operator[](const int i)
  {
    return values[i];
  }


  friend std::istream &operator>>(std::istream &is, ParseVector<T> &v)
  {
    for (size_t i = 0; i < v.values.size(); ++i)
      is >> v.values[i];
    return is;
  }
};
