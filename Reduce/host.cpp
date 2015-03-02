//Use cl::vector instead of STL version
#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace cl;

#include <oclUtils.hpp>

int main() {

	try {
#pragma region Initialize GPU
		Context context;
		if (!oclCreateContextBy(context, "nvidia")) {
			throw Error(CL_INVALID_CONTEXT, "Failed to create a valid context!");
		}

		// Query devices from the context
		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// Create a command queue and use the first device
		CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		// Read source file
		auto sourceCode = oclReadSourcesFromFile("reduce_kernel.cl");
		Program::Sources sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		// Make program of the source code in the context
		Program program(context, sources);

		// Build program for these specific devices
		try {
			program.build(devices);
		}
		catch (Error error) {
			oclPrintError(error);
			std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
			std::cerr << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
			std::cerr << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			throw error;
		}

		// Make kernel
		Kernel kernel(program, "reduce");
#pragma endregion

		for (unsigned size = 1 << 23; size <= 1 << 27; size = size << 1) {
			std::vector<int> input(size);
			std::default_random_engine rand(0);
			std::generate(input.begin(), input.end(), [&] () {return rand() % 512 - 256; });

			auto CPU_result = std::accumulate(input.begin(), input.end(), 0);

#pragma region Execute kernel
			// Create memory buffers
			Buffer input_buffer(context, CL_MEM_READ_ONLY, size * sizeof(int));
			Buffer result_buffer(context, CL_MEM_READ_WRITE, sizeof(int));

			int GPU_result = 0;
			// Copy input to the memory buffer
			queue.enqueueWriteBuffer(result_buffer, CL_TRUE, 0, sizeof(int), &GPU_result);
			queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, size * sizeof(int), input.data());
			queue.finish(); // NEW: Ez néhány platformon lehet hogy megold egy biz. problémát

			// Set arguments to kernel
			kernel.setArg(0, input_buffer);
			kernel.setArg(1, result_buffer);

			// Run the kernel on specific ND range
			Event operation;
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, size, cl::NullRange, nullptr, &operation);

			operation.wait();
			cl_ulong device_start, device_end;
			operation.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &device_start);
			operation.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &device_end);
			auto time = (device_end - device_start) / 1000000;

			// Read buffer into a local variable
			queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int), &GPU_result);
			queue.finish(); // NEW: Ez néhány platformon lehet hogy megold egy biz. problémát
#pragma endregion

			std::cout << time << std::endl;
			
			if (CPU_result != GPU_result) {
				std::cerr << "computation error" << std::endl;
				break;
			}
		}

	}
	catch (Error error) {
		oclPrintError(error);
	}

	std::cin.get();

	return 0;
}