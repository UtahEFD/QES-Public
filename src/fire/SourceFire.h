/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file SourceFire.h
 * @brief This class specifies Fire sources for QES-Fire and QES-Plume integration
 */

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
