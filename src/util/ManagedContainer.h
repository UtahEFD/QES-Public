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

/** @file ManagedContainer.h
 */

#pragma once

#include <iostream>
#include <vector>
#include <queue>

template<class T>
class ManagedContainer
{
protected:
  // size_t nbr_used = 0;
  size_t nbr_inserted = 0;

  std::vector<T> elements;
  std::vector<size_t> added;
  std::queue<size_t> available;

private:
  size_t scrub_elements()
  {
    size_t nbr_used = 0;
    added.clear();
    std::queue<size_t> empty;
    std::swap(available, empty);

    for (size_t it = 0; it < elements.size(); ++it) {
      if (elements[it].isActive)
        nbr_used++;
      else
        available.push(it);
    }
    return nbr_used;
  }

public:
  ManagedContainer() = default;
  explicit ManagedContainer(size_t n) : elements(n)
  {
    for (size_t k = 0; k < n; ++k)
      available.push(k);
  }
  ~ManagedContainer() = default;

  size_t front() { return available.front(); }
  size_t size() { return elements.size(); }
  size_t get_nbr_added() { return added.size(); }
  size_t get_nbr_inserted() { return nbr_inserted; }
  size_t get_nbr_active()
  {
    size_t nbr_used = 0;
    for (auto &p : elements) {
      if (p.isActive)
        nbr_used++;
    }
    return nbr_used;
  }

  T *last_added()
  {
    return &elements[added.back()];
  }
  T *get_added(const size_t k)
  {
    return &elements[added[k]];
  }

  T *get(const size_t k)
  {
    return &elements[k];
  }

  typename std::vector<T>::iterator begin()
  {
    return elements.begin();
  }

  typename std::vector<T>::iterator end()
  {
    return elements.end();
  }

  bool check_size(const int &needed)
  {
    return needed <= available.size();
  }

  void sweep(const int &new_part)
  {
    size_t nbr_used = scrub_elements();
    size_t elements_size = elements.size();
    if (elements_size < nbr_used + new_part) {
      elements.resize(elements_size + new_part);
      for (size_t it = elements_size; it < elements.size(); ++it) {
        available.push(it);
      }
    }
  }

  void insert()
  {
    elements[available.front()] = T(nbr_inserted);
    nbr_inserted++;
    // elements[available.front()].isActive = true;
    added.emplace_back(available.front());
    available.pop();
  }
};