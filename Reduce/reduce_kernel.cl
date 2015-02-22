__kernel void reduce(__global int* idata, __global int* odata, __global const int* offset) {
	int id = get_global_id(0);
	if (id % *offset == 0 && id + offset < get_global_size(0)) {
		odata[id] = idata[id] + idata[id + *offset];
	}
}

// id % (*offset*2) (logical error)
// id + *offset		(programming mistake)