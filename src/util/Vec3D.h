#pragma once

class Vec3D
{
private:
  float d[3];

public:
  Vec3D() { d[0] = d[1] = d[2] = 0.0f; }
  Vec3D(float x, float y, float z)
  {
    d[0] = x;
    d[1] = y;
    d[2] = z;
  }
  ~Vec3D() {}

  float operator[](const int idx) const
  {
    return d[idx];
  }

  float &operator[](const int idx)
  {
    return d[idx];
  }
};
