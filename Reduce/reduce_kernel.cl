__kernel void hybrid_reduce(__global int *input, const unsigned length, __global int *result, __local int *scratch) {

	int l_id = get_local_id(0);
	int g_id = get_global_id(0);

	int sum = 0;

	// Loop sequentially over chunks of input vector
	while (g_id < length) {
		sum += input[g_id];
		g_id += get_global_size(0);
	}
	scratch[l_id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform parallel reduction
	for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
		if (l_id < offset) {
			scratch[l_id] += scratch[l_id + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (l_id == 0) {
		result[get_group_id(0)] = scratch[0];
	}
}

__kernel void parallel_reduce(__global int *input, __global int *result, __local int *scratch) {

	int l_id = get_local_id(0);

	scratch[l_id] = input[l_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform parallel reduction
	for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
		if (l_id < offset) {
			scratch[l_id] += scratch[l_id + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (l_id == 0) {
		*result = scratch[0];
	}
}