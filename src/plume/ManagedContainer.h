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

/** @file ManagedContainer.h
 */

#pragma once

#include <iostream>
#include <vector>
#include <queue>

#include "Particle.h"

template<class T>
class ManagedContainer
{
protected:
  // size_t nbr_used = 0;
  size_t nbr_inserted = 0;

  std::vector<T> elements;
  std::vector<size_t> added;
  std::queue<size_t> available;

  size_t m_resized = 0;

private:
  size_t screen_elements()
  {
    size_t nbr_used = 0;
    added.clear();
    std::queue<size_t> empty;
    std::swap(available, empty);

    for (size_t it = 0; it < elements.size(); ++it) {
      if (elements[it].state == ACTIVE)
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

  size_t size() { return elements.size(); }
  size_t get_nbr_added() { return added.size(); }
  size_t get_nbr_inserted() { return nbr_inserted; }
  size_t get_nbr_active()
  {
    size_t nbr_used = 0;
    for (auto &p : elements) {
      if (p.state == ACTIVE)
        nbr_used++;
    }
    return nbr_used;
  }

  T *last_added() { return &elements[added.back()]; }
  T *get_added(const size_t &k) { return &elements[added[k]]; }
  size_t get_added_index(const size_t &k) { return added[k]; }
  T *get_ptr(const size_t &k) { return &elements[k]; }
  T &at(const size_t &k) { return elements[k]; }
  T &operator[](const size_t &k) { return elements[k]; }

  typename std::vector<T>::iterator begin() { return elements.begin(); }
  typename std::vector<T>::iterator end() { return elements.end(); }

  bool check_size(const int &needed) { return needed <= available.size(); }

  void check_resize(const int &new_part)
  {
    m_resized = 0;
    size_t nbr_used = screen_elements();
    size_t elements_size = elements.size();
    if (elements_size < nbr_used + new_part) {
      m_resized = new_part;
      elements.resize(elements_size + new_part);
      for (size_t it = elements_size; it < elements.size(); ++it) {
        available.push(it);
      }
    }
  }

  template<typename U>
  void resize_companion(std::vector<U> &v)
  {
    if (m_resized > 0) {
      size_t elements_size = v.size();
      v.resize(elements_size + m_resized);
      if (elements.size() != v.size())
        throw std::runtime_error("[ManagedContainer] mismatch containers sizes");
    }
  }

  void insert()
  {
    // reset the particle:
    // - turning the particle active:
    elements[available.front()].state = ACTIVE;
    // adding index to added and remove from available:
    added.emplace_back(available.front());
    available.pop();
    // increment the counter
    nbr_inserted++;
  }

  void obtain_available(const size_t &n, std::vector<size_t> &out)
  {
    out.clear();
    out.resize(n);
    // reset the particle:
    // - turning the particle active:
    // adding index to out and remove from available:
    for (size_t k = 0; k < n; ++k) {
      out[k] = available.front();
      available.pop();
    }
    // adding index to added
    added.insert(added.end(), out.begin(), out.end());
    // increment the counter
    nbr_inserted += n;
  }
};
