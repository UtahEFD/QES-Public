/*
 * copy_turbulence.cu
 * This file is part of CUDAPLUME
 *
 * Copyright (C) 2012 - Alex
 *
 * CUDAPLUME is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * CUDAPLUME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
 */

 #ifndef __COPY_TURBULENCE_CU_H__
 #define __COPY_TURBULENCE_CU_H__
 
#include "particles_kernel.cu"

 struct copy_turb_functor
 {
   template <typename Tuple>
  __device__
  void operator()(Tuple t)
  {
//     device_turbs
  }
   
 }
 
 #endif /* __COPY_TURBULENCE_CU_H__ */
 
