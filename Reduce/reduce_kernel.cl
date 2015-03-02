__kernel void reduce(__global int *input, const unsigned length, __global int *result, __local int *scratch) {

	// each thread loads one element from global to local mem
	int l_id = get_local_id(0);
	int g_id = get_global_id(0);

	scratch[l_id] = g_id < length ? input[g_id] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in shared mem
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