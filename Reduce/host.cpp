#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace std;

int main() {

	for (long long size = 1 << 23; size <= 1 << 28; size = size << 1) {
		vector<int> input(size);
		default_random_engine rand(0);
		generate(input.begin(), input.end(), [&] () {return rand() % 512 - 256; });

		auto start = std::chrono::high_resolution_clock::now();
		auto result = accumulate(input.begin(), input.end(), 0);
		auto done = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(done - start).count();
		cout << time << endl;

		//without the side-effect the compiler would optimize the whole accumulate out
		if (result > size * 256) {return 0;}
	}
	cin.get();
	return 0;
}