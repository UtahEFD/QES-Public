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

/** @file Edge.h */

#pragma once

#include "util/Vector3.h"
#include <vector>

/**
 * @class Edge
 * @brief Contains 2 Vector3s and represents a connection
 * between these two points.
 */
template<class T>
class Edge
{
private:


public:
  std::vector<T> values;
  /**
   * Default constructor, both items initialized as 0.
   */
  Edge();

  /**
   * This constructor takes in two values and sets them
   * as values 0 and 1 respectively.
   *
   * @param a value to be assigned to point 0
   * @param b value to be assigned to point 1
   */
  Edge(const T a, const T b);

  /**
   * Operator overload of the [] operator. This
   * is how data members are accessed. Value of the
   * input is sanitized for array out of bounds errors.
   *
   * @param i index indicating which point should be returned
   * @return a reference to the value denoted by i
   */
  T &operator[](const int i);

  /**
   * == comparative operator overload.
   * this returns true if the values in each edge match.
   *
   * @note if the values on the edges are swapped this is still true.
   * @param e edge to be compared to this edge
   * @return true if edges are equal, else false
   */
  bool operator==(const Edge<T> e) const;

  /**
   * Checks to see if value v exists in this edge. If it does,
   * then it returns the index of that value. If it doesn't this
   * returns -1.
   *
   * @param v value to query for index
   * @return the index of the given value, -1 if not found
   */
  int getIndex(T v);
};

// this is because this is a template class
#include "Edge.cpp"
