__kernel void reduce(__global int *input, __global int *result, __local int *scratch) {

	// each thread loads one element from global to local mem
	int l_id = get_local_id(0);
	int g_id = get_global_id(0);
	scratch[l_id] = input[g_id];
	result[g_id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in shared mem
	for (int s = 1; s < get_local_size(0); s *= 2) {
		if (l_id % (2*s) == 0) {
			scratch[l_id] += scratch[l_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (l_id == 0) {
		result[get_group_id(0)] = scratch[0];
	}
}