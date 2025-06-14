/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

struct cutVert
{
  cutVert()
    : x_cut(0.0), y_cut(0.0), z_cut(0.0) {}

  cutVert(float x, float y, float z)
    : x_cut(x), y_cut(y), z_cut(z) {}

  float x_cut, y_cut, z_cut;
};

struct cutCell
{
  cutCell(int id)
    : cell_id(id), pass_number(1), z_solid(0.0),
      ni(0.0), nj(0.0), nk(0.0),
      s_behind(0.0), s_front(0.0), s_right(0.0), s_left(0.0), s_above(0.0), s_below(0.0),
      corner_id(8, 1) {}

  int cell_id;
  int pass_number;
  float z_solid;
  float ni, nj, nk;
  float s_behind, s_front, s_right, s_left, s_above, s_below;
  std::vector<cutVert> face_behind;
  std::vector<cutVert> face_front;
  std::vector<cutVert> face_right;
  std::vector<cutVert> face_left;
  std::vector<cutVert> face_above;
  std::vector<cutVert> face_below;
  std::vector<int> corner_id;
  std::vector<cutVert> intersect;
};
