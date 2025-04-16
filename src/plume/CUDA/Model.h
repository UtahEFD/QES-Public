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

/** @file
 * @brief
 */

#ifndef __CUDA_MODEL_H__
#define __CUDA_MODEL_H__

#include "plume/ParticleIDGen.h"

#include "plume/CUDA/QES_data.h"
#include "plume/CUDA/Interpolation.h"
#include "plume/CUDA/Partition.h"
#include "plume/CUDA/RandomGenerator.h"

typedef struct
{
  float xStartDomain;
  float yStartDomain;
  float zStartDomain;

  float xEndDomain;
  float yEndDomain;
  float zEndDomain;

} BC_Params;

class Model
{
public:
  Model()
  {
    id_gen = ParticleIDGen::getInstance();
  }

  ~Model()
  {
  }

  void getNewParticle(const int &num_new_particle,
                      particle_array d_particle,
                      const QESTurbData &d_qes_turb_data,
                      const QESgrid &qes_grid,
                      RandomGenerator *random,
                      Interpolation *interpolation,
                      Partition *partition);

  void advectParticle(particle_array d_particle,
                      const int &num_new_particle,
                      const BC_Params &bc_param,
                      RandomGenerator *random);

private:
  ParticleIDGen *id_gen;
};

#endif
