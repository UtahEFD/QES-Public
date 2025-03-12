# 
# OptiX Macros
# -- needed to compile OptiX code
# 

# Function to compile and embed PTX files using bin2c
function(cuda_compile_and_embed output_var cuda_file ptx_name)

  # Define the PTX file name
  set(ptx_file ${CMAKE_BINARY_DIR}/ptx/${ptx_name}.ptx)

  # Add a custom command to compile CUDA code to PTX
  add_custom_command(
    OUTPUT ${ptx_file}
    COMMAND ${CUDA_NVCC_EXECUTABLE} -I${OptiX_INCLUDE} -I${CMAKE_SOURCE_DIR}/src -o ${ptx_file} -ptx ${cuda_file}
    DEPENDS ${cuda_file}
    COMMENT "Compiling ${cuda_file} to PTX"
    )

  # Define the embedded C file name
  set(embedded_file ${CMAKE_BINARY_DIR}/ptx/${ptx_name}_embedded.h)

  # Add a custom command to convert PTX to C source using bin2c
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${ptx_name}_ptx ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "Converting ${ptx_file} to C++ header"
    )

  # Set the output variable
  set(${output_var} ${embedded_file} PARENT_SCOPE)
endfunction()


