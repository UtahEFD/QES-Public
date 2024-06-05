/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file OptixRayTrace.h
 *
 * OptiX version of MixingLength
 * OptiX Version: 7.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <optix_stubs.h>

#include <stdexcept>
#include <sstream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
#include <assert.h>
#include <typeinfo>
#include <iostream>
#include <fstream>

#include "util/Triangle.h"

#define RAY_TYPE_COUNT 2
#define RAY_TYPE_RADIENCE 0


#define CUDA_CHECK(call)                                 \
  do {                                                   \
    cudaError_t error = call;                            \
    if (error != cudaSuccess) {                          \
      std::stringstream strStream;                       \
      strStream << "CUDA call ( " << #call               \
                << " ) failed with error: '"             \
                << cudaGetErrorString(error)             \
                << "' (" << __FILE__ << ":"              \
                << __LINE__ << ")\n";                    \
      throw std::runtime_error(strStream.str().c_str()); \
    }                                                    \
  } while (0);


#define CUDA_SYNC_CHECK()                                  \
  do {                                                     \
    cudaDeviceSynchronize();                               \
    cudaError_t error = cudaGetLastError();                \
    if (error != cudaSuccess) {                            \
      std::stringstream strStream;                         \
      strStream << "CUDA error on synchronize with error " \
                << cudaGetErrorString(error)               \
                << " (" __FILE__ << ":"                    \
                << __LINE__ << ")\n";                      \
      throw std::runtime_error(strStream.str().c_str());   \
    }                                                      \
  } while (0)

#define OPTIX_CHECK(call)                                \
  do {                                                   \
    OptixResult res = call;                              \
    if (res != OPTIX_SUCCESS) {                          \
      std::stringstream strStream;                       \
      strStream << optixGetErrorName(res) << ":"         \
                << "Optix call ( " << #call              \
                << " ) failed: " __FILE__ ":"            \
                << __LINE__ << "\n";                     \
      throw std::runtime_error(strStream.str().c_str()); \
    }                                                    \
  } while (0)


/**
 * Struct for the sbt records.
 */
template<typename T>
struct Record
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};


/**
 * Struct for holding 3 float values
 * will be replaced in the near future.
 */
struct Vertex
{
  float x;
  float y;
  float z;

  friend std::ostream &operator<<(std::ostream &os, const Vertex &v)
  {
    os << "<" << v.x << ", " << v.y << ", " << v.z << ">";
    return os;
  };
};


/**
 * Struct to hold values returned from OptiX launches.
 * Values from payloads in the .cu file can be added here.
 */
struct Hit
{
  float t;
};


/**
 * Struct containing variables passed to and from device and host.
 * Any variable shared between device and host can be added here.
 */
struct Params
{
  OptixTraversableHandle handle;
  Hit *hits;
  int *icellflagArray;// change to this later
  float dx, dy, dz;
  int numSamples;
};


/**
 * Struct to contain OptiX raygen data that can be stored in sbt record.
 * None needed in mixing length case.
 */
struct RayGenData
{
  // no data needed
};


/**
 * Struct to contain OptiX miss data that can be stored in sbt record.
 * None needed in mixing length case.
 */
struct MissData
{
  // no data needed
};


/**
 * Struct to contain OptiX closest hit data that can be stored in sbt record.
 *
 * None needed in mixing length case.
 */
struct HitGroupData
{
  // no hit data needed
};


/**
 * Struct to contain OptiX ray tracing data.
 * Represents one ray tracing state in which all related variables to
 * OptiX construction are stored.
 */
struct RayTracingState
{
  OptixDeviceContext context = 0;// stores device context
  CUstream stream = 0;// CUDA stream

  OptixModule ptx_module = 0;// OptiX module containing .cu functions

  OptixPipeline pipeline = 0;// OptiX shared pipeline
  OptixPipelineCompileOptions pipeline_compile_options = {};// shared pipeline options


  OptixTraversableHandle gas_handle = 0;// ptr to AS
  CUdeviceptr d_gas_output_buffer = 0;
  CUdeviceptr d_tris;// converted mesh list

  // program groups related to OptiX
  OptixProgramGroup raygen_prog_group = 0;
  OptixProgramGroup miss_prog_group = 0;
  OptixProgramGroup hit_prog_group = 0;

  Params params = {};// portal between device and host

  CUdeviceptr icellflagArray_d;// ptr to icellflag array to pass to device

  CUdeviceptr outputBuffer = 0;// buffer to read and write btw device and host
  CUdeviceptr paramsBuffer = 0;// buffer to read parms info btw device and host

  OptixShaderBindingTable sbt = {};// OptiX


  // variables passed in from mixing length function
  int nx, ny, nz;
  float dx, dy, dz;
};

/**
 * @class OptixRayTrace
 * @brief :document this:
 *
 * long desc here
 */
class OptixRayTrace
{
public:
  /**
   * Initializes OptiX and creates the context.
   *
   * If not testing, the acceleration structure will be based off of the
   * provided list of Triangle objects.
   *
   * @param tris List of Triangle objects.
   */
  OptixRayTrace(std::vector<Triangle *> tris);

  ~OptixRayTrace();


  /**
   * Calculates the mixing length.
   *
   * @param numSamples The probablistic sampling of per-cell launch
   *        directions
   * @param dimX Domain info in the x plane
   * @param dimY Domain info in the y plane
   * @param dimZ Domain info in the z plane
   * @param dx Grid info in the x plane
   * @param dy Grid info in the y plane
   * @param dz Grid info in the z plane
   * @param icellflag Cell type
   * @param mixingLengths Array of mixinglengths for all cells that will be updated
   */
  void calculateMixingLength(int numSamples, int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag, std::vector<double> &mixingLengths);

private:
  OptixRayTrace();// cannot have an empty constructor (have to pass in a mesh to build)


  RayTracingState state;// needs to be accessable though out program


  // Cuda device context and & properties that pipeline will run on
  CUcontext cudaContext; /**< CUDA device context */
  cudaDeviceProp deviceProps; /**< CUDA device properties */

  ///@{
  /** OptiX sbt records */
  typedef Record<RayGenData> RayGenRecord;
  typedef Record<MissData> MissRecord;
  typedef Record<HitGroupData> HitGroupRecord;
  ///@}

  std::string ptx; /**< Holds ptx string */

  /**
   * Initializes OptiX and confirms OptiX compatible devices are present.
   *
   * @throws RuntimeException On no OptiX 7.0 compatible devices
   */
  void initOptix();


  /**
   * Builds acceleration structure with provided list of Triangle objects.
   *
   * @note Non-test version of AS.
   * @param tris List of Triangle objects representing given terrain and buildings.
   */
  void buildAS(std::vector<Triangle *> tris);


  /**
   * Builds acceleration structure with 2 Triangles representing the
   * ground of the domain.
   *
   * @note Test version of AS
   * @note If on test version, a message will be printed to terminal in red text.
   */
  void buildAS();


  /**
   * Creates and configures a optix device context for primary GPU device.
   */
  void createContext();


  /**
   * Creates OptiX module from generated ptx file.
   */
  void createModule();


  /**
   * Creates OptiX program groups.
   * Three groups: raygen, miss, and closest hit.
   */
  void createProgramGroups();


  /**
   * Creates OptiX pipeline.
   */
  void createPipeline();


  /**
   * Creates OptiX SBT record.
   */
  void createSBT();


  /**
   * Launches OptiX.
   * Starts the ray launching process.
   */
  void launch();


  /**
   * Frees up memory from state variables.
   */
  void cleanState();


  /**
   * Converts the list of Traingle objects to a list of Vertex objects.
   * This is for the purpose of OptiX and to not conflict with other
   * parts of the code.
   *
   * @param tris The list of Triangle objects representing the buildings
   *        and terrain.
   * @param trisArray The list of Vertex struct objects representing the
   *        converted list of Triangles.
   */
  void convertVecMeshType(std::vector<Triangle *> &tris, std::vector<Vertex> &trisArray);


  /**
   * Initialize members of the Params stuct for state.params.
   *
   * @param dimX Domain info in the x plane
   * @param dimY Domain info in the y plane
   * @param dimZ Domain info in the z plane
   * @param dx Grid info in the x plane
   * @param dy Grid info in the y plane
   * @param dz Grid info in the z plane
   * @param icellflag Cell type
   */
  void initParams(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag);


  /**
   * Helper function for rounding up.
   * Used in accelerated structure creation.
   *
   * @param x :document this:
   * @param y :document this:
   */
  size_t roundUp(size_t x, size_t y);
};
