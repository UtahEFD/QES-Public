CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

CUDA_ADD_LIBRARY(UrbKernels
  abs_diff.cu
	comp_denoms.cu 
  cudaMatrixIO.cu
  deviceInfo.cu
	diffusion.cu 
	divergence.cu 
	euler.cu
	iteration_red_black_comp_v2.cu 
  map_matrix_to_interval.cu
  max.cu
  min.cu
  rel_diff.cu
	setup_bndry_mats.cu 
  sum.cu
	turbulence.cu
  zero.cu
)		

