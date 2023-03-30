// Parallel pattern: HISTOGRAM.
kernel void hist_simple(global const unsigned char* A, global int* H, const int n_bins) {
	int id = get_global_id(0);
	int bin_index = A[id];
	float bin_factor = (float)n_bins / (float)256;
	int idx = bin_factor * (float)bin_index;  // float --> int conversion
	atomic_inc(&H[idx]);
}
//if (bin_index > n_bins - 1) { bin_index = n_bins - 1; }
//atomic_inc(&H[bin_index]);

/*
kernel void hist_simple(global const unsigned char* A, global int* H, const int n_bins) {
	int id = get_global_id(0);
	int bin_index = A[id];
	//atomic_inc(&H[bin_index]);
}*/

// Parallel pattern: SCAN.
kernel void hist_cumulative(global const int* H, global int* HC) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int a = id + 1; a < N; a++) {
		atomic_add(&HC[a], H[id]);
	}
}


kernel void normalise(global int* HC, global int* LUT, const int n_bins) {
	int id = get_global_id(0);
	int hc_id = id;
	if (hc_id > n_bins-1) {
		hc_id = n_bins-1;
	}
	float factor = (float)(255) / HC[n_bins-1];
	LUT[id] = HC[hc_id] * factor;
}


// Parallel pattern: MAP.
kernel void back_projection(global const unsigned char* A, global const int* LUT, global unsigned char* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]];
}


// From Tutorial 3.
// Blelloch basic exclusive scan.
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;
	// Up-sweep.
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];
		// Sync the step.
		barrier(CLK_GLOBAL_MEM_FENCE); 
	}

	// Down-sweep.
	if (id == 0)
		A[N - 1] = 0;  // Exclusive scan.
	// Sync the step.
	barrier(CLK_GLOBAL_MEM_FENCE); 

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride];  // Reduce. 
			A[id - stride] = t;		  // Move.
		}
		// Sync the step.
		barrier(CLK_GLOBAL_MEM_FENCE); 
	}
}


// From Tutorial 3.
// Hillis-Steele basic inclusive scan.
// Requires additional buffer B to avoid data overwrite.
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];
		// Sync the step.
		barrier(CLK_GLOBAL_MEM_FENCE); 
		// Swap A & B between steps.
		C = A; A = B; B = C; 
	}
}


// To delete - use CImg in-built function.
// https://www.programmingalgorithms.com/algorithm/rgb-to-ycbcr/cpp/
kernel void rgb2ycbcr(global const unsigned char* A, global unsigned char* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3;  // Each image consists of 3 colour channels.
	int cc = id / image_size;  // colour channel: 0 - red, 1 - green, 2 - blue.

	/*
	float fr = (float)rgb.R / 255;
	float fg = (float)rgb.G / 255;
	float fb = (float)rgb.B / 255;

	float Y = (float)(0.2989 * fr + 0.5866 * fg + 0.1145 * fb);
	float Cb = (float)(-0.1687 * fr - 0.3313 * fg + 0.5000 * fb);
	float Cr = (float)(0.5000 * fr - 0.4184 * fg - 0.0816 * fb);
	*/



	// Offsets when in each colour channel to get to other channels (RGB).
	// e.g. when in GREEN channel:
	//     - 1 * image_size to get to red,
	//       0 * image_size to stay in green,
	//     + 1 * image-size to get to blue.
	int offsets[3][2] = { { 0,  1,  2},    // Red.
							{-1,  0,  1},  // Green.
							{-2, -1, 0}    // Blue.
	};

	float multipliers[3][2] = { { 0.2989, -0.1687,  0.5000},  // Red multipliers for R,G,B.
								{ 0.5866, -0.3313, -0.4184},  // Green multipliers for R,G,B.
								{ 0.1145,  0.5000, -0.0816}   // Blue multipiers for R,G,B.
	};

	for (int a = 0; a < 3; a++) {
		B[id + offsets[cc][0] * image_size] += multipliers[cc][0] * (A[id] / 255);
	}

	// THIS IS WRONG
	// SEE WEBSITE
	// MULTIPLIERS ETC MUST JUST DEPENDING ON MEMORY ADDRESS
	// 
	// 	// Multiplier for RGB values to make YCbCr.
	//float red_m[3] = { 0.2989, 0.5866 , 0.1145 };  // RGB multipliers
	//float green_m[3] = { -0.1687, -0.3313, 0.5000 };
	//float blue_m[3] = { 0.5000 , -0.4184, -0.0816 };
	// 
	//B[id] += multiplier[cc] * A[id];
	//B[id + offsets[cc][1] * image_size] += multiplier[cc] * A[id];
	//B[id + offsets[cc][2] * image_size] += multiplier[cc] * A[id];
}
