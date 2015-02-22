__kernel void reduce(__global const int* input, __global int* result) {
	result += input[get_global_id(0)];
}