/*
 *particles_kernel.cuh
 * This file is part of CUDAPLUME
 *
 * Copyright (C) 2012 - Alex Geng
 *
 * CUDAPLUME is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CUDAPLUME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
 */
/* 
 * CUDA particle kernel code.
 */

#ifndef _PARTICLES_KERNEL_CUH_
#define _PARTICLES_KERNEL_CUH_
#include <cutil_math.h> 

texture<float4, 3, cudaReadModeElementType> windTex;    

#endif
