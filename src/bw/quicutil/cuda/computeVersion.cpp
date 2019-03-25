/**
 * File: computeVersion.cpp
 * Author: Joshua Clark
 * Date: Wednesday, September 17th, 2010
 *
 * This program outputs the lowest CUDA compute version of the available 
 * graphics hardware. If no compatible CUDA hardware is found, the program will
 * output "00". The format of the output is two digits "XY" representing the
 * compute version "X.Y".
 *
 * Caveat: When running on a MacBook Pro with automatic graphics switching, you
 * must have automatic graphics switching disabled for this program to correctly
 * probe the devices.
 */

#include <iostream>
#include <cuda_runtime_api.h>

int main(int argc, char ** argv) {
	
	// Initialize the default compute version to no compatible CUDA hardware.
	int compute_major = 9999;
	int compute_minor = 9999;
	
	// Probe the device count, if there was an error output that no compatible
	// CUDA hardware was found.
	int device_count = 0;
	if(cudaGetDeviceCount(&device_count) != cudaSuccess) {
		std::cout << compute_major << compute_minor << std::endl;
		return 1;
	}
	
	// Loop through each of the CUDA compatible devices on the system.
	for(int device_id = 0; device_id < device_count; device_id++) {
		
		// Get the properties of the current device.
		cudaDeviceProp device_properties;
		cudaGetDeviceProperties(&device_properties, device_id);
		
		// If the current CUDA version is lower than the current, than use this
		// version.
		if (device_properties.major < compute_major) {
			compute_major = device_properties.major;
			compute_minor = device_properties.minor;
		} else if(device_properties.major == compute_major && device_properties.minor < compute_minor) {
			compute_minor = device_properties.minor;
		}
		
	}
	
	// Print the compute version.
	std::cout << compute_major << compute_minor << std::endl;
	
	return 0;
}
