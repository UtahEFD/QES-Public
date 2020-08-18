    	cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    	project(SOR_global)


	set(CUDA_SEPARABLE_COMPILATION ON)
	# CUDA PACKAGE
	find_package(CUDA)
	set(CUDA_SEPARABLE_COMPILATION ON)
	link_directories( "/usr/local/cuda-9.1")
	target_link_libraries(cudadevrt.lib )
	#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	#set(CUDA_HOST_COMPILER gcc)
	#set(CUDA_SDK_ROOT_DIR "/usr")
	#set(CUDA_TOOLKIT_ROOT_DIR=/usr)

	# COMPILE CU FILES

	list( APPEND CUDA_NVCC_FLAGS "-arch=sm_61;-rdc=true;-std=c++11")

	# COMPILE AND LINK
	CUDA_ADD_EXECUTABLE(SOR SOR_dynamic.cu)







