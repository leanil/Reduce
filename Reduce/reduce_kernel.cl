__kernel void reduce(__global int* input, __global int* result) {
	int n = get_global_size(0);
	int x = get_global_id(0);
	for (int offset = 1; offset < n; offset = offset << 1) {
		if (x  % 2*offset == 0 && x + offset < n) {
			input[x] += input[x + offset];
		}
	}
	*result = input[0];
}