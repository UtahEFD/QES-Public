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
    : cell_id(id), corner_id(8, 1), z_solid(0.0), pass_number(1),
      ni(0.0), nj(0.0), nk(0.0), s_behind(0.0), s_front(0.0), s_right(0.0),
      s_left(0.0), s_below(0.0), s_above(0.0) {}

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
