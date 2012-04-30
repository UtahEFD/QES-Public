CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

CUDA_ADD_LIBRARY(UrbKernels
	comp_denoms.cu 
	divergence.cu 
	diffusion.cu 
	euler.cu
	iteration_red_black_comp_v2.cu 
	setup_bndry_mats.cu 
	turbulence.cu
)		
