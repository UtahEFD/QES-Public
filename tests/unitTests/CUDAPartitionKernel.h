#pragma once

#include <cuda.h>
#include <chrono>
#include <cstdio>

void partitionData(std::vector<float> &arr, float pivot);
