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

/** @file Edge.cpp */

template<class T>
Edge<T>::Edge()
{
  values.push_back(0);
  values.push_back(0);
}

template<class T>
Edge<T>::Edge(const T a, const T b)
{
  values.push_back(a);
  values.push_back(b);
}

template<class T>
T &Edge<T>::operator[](const int i)
{
  return values[i % 2];
}

template<class T>
bool Edge<T>::operator==(const Edge<T> e) const
{
  if (values[0] == e.values[0])
    return values[1] == e.values[1];
  else if (values[0] == e.values[1])
    return values[1] == e.values[0];
  else
    return false;
}

template<class T>
int Edge<T>::getIndex(T v)
{
  if (values[0] == v)
    return 0;
  else if (values[1] == v)
    return 1;
  else
    return -1;
}
