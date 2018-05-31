    	cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    	project(SOR_global)


	# CUDA PACKAGE
	find_package(CUDA)

	# COMPILE CU FILES

	list( APPEND CUDA_NVCC_FLAGS "-arch=sm_61;-std=c++11")

	# COMPILE AND LINK
	CUDA_ADD_EXECUTABLE(SOR SOR_Global.cu)
