# The CUDA_ADD_LIBRARY automatically pulls in the necessary header
# paths for finding CUDA related includes and since this is needed for
# some of the files below, the CUDA_ADD_LIBRARY macro will be used.
CUDA_ADD_LIBRARY(CommonKernels
	abs_diff.cu
	map_matrix_to_interval.cu
	max.cu
	min.cu
	pack.cu
  rel_diff.cu
	sum.cu
	zero.cu
	showerror.cu
)		
