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
#include <math.h>

using namespace cl;

#include <oclUtils.hpp>

unsigned round_up_div(unsigned a, unsigned b) {
	return static_cast<int>(ceil((double)a / b));
}

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
		Kernel kernel_0(program, "reduce"), kernel_1(program, "reduce");
#pragma endregion

		for (unsigned size = 1 << 20; size <= 1 << 27; size = size << 1) {
			std::vector<int> input(size);
			std::default_random_engine rand(0);
			std::generate(input.begin(), input.end(), [&] () {return rand() % 512 - 256; });

			auto CPU_result = std::accumulate(input.begin(), input.end(), 0);

#pragma region Execute kernel
			// Create memory buffers
			Buffer io_buffer_0(context, CL_MEM_READ_WRITE, size * sizeof(int));
			Buffer io_buffer_1(context, CL_MEM_READ_WRITE, size * sizeof(int));

			// Copy input to the memory buffer
			queue.enqueueWriteBuffer(io_buffer_0, CL_TRUE, 0, size * sizeof(int), input.data());
			queue.finish(); // NEW: Ez néhány platformon lehet hogy megold egy biz. problémát

			cl_ulong round = 0;
			const unsigned GROUP_SIZE = 128;

			kernel_0.setArg(0, io_buffer_0);
			kernel_0.setArg(2, io_buffer_1);
			kernel_0.setArg(3, GROUP_SIZE * sizeof(int), nullptr);

			kernel_1.setArg(0, io_buffer_1);
			kernel_1.setArg(2, io_buffer_0);
			kernel_1.setArg(3, GROUP_SIZE * sizeof(int), nullptr);

			auto start = std::chrono::high_resolution_clock::now();

			for (unsigned rem_size = size; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE), ++round) {

				if (round % 2 == 0) {
					kernel_0.setArg(1, rem_size);
				}
				else {
					kernel_1.setArg(1, rem_size);
				}
				// Run the kernel on specific ND range
				queue.enqueueNDRangeKernel(round % 2 == 0 ? kernel_0 : kernel_1,
					NullRange, round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE, GROUP_SIZE);
			}

			queue.finish();
			auto stop = std::chrono::high_resolution_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

			// Read buffer into a local variable
			int GPU_result;
			queue.enqueueReadBuffer(round % 2 == 0 ? io_buffer_0 : io_buffer_1, CL_TRUE, 0, sizeof(int), &GPU_result);
			queue.finish(); // NEW: Ez néhány platformon lehet hogy megold egy biz. problémát
#pragma endregion

			std::cout << time << std::endl;

			if (CPU_result != GPU_result) {
				std::cerr << "computation error" << std::endl;
			}
		}

	}
	catch (Error error) {
		oclPrintError(error);
	}

	std::cin.get();

	return 0;
}