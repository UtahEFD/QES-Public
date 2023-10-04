//
// Created by Fabien Margairaz on 10/3/23.
//

#ifndef QES_SOURCEFIRE_H
#define QES_SOURCEFIRE_H

#include "plume/Source.hpp"

class SourceFire : public Source
{
private:
  SourceFire() : Source()
  {}

protected:
  float x, y, z;
  int particle_per_time;
  bool active = true;

public:
  SourceFire(const float &x_in, const float &y_in, const float &z_in, const int &pp_in)
    : Source(), x(x_in), y(y_in), z(z_in), particle_per_time(pp_in)
  {
    sourceIdx = 0;
  }
  ~SourceFire() = default;

  // this function should be customized based on the fire code.
  void setSource() {}

  virtual int emitParticles(const float &dt,
                            const float &currTime,
                            std::list<Particle *> &emittedParticles);
};


#endif// QES_SOURCEFIRE_H
