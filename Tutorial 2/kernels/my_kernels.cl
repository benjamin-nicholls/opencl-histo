// A simple OpenCL kernel which copies all pixels from A to B.
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}


kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3;  // Each image consists of 3 colour channels.
	int colour_channel = id / image_size;  // 0 - red, 1 - green, 2 - blue.

	if (colour_channel == 0) {
		B[id] = A[id];
	} else {
		B[id] = 0;
	};
}


kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3;  // Each image consists of 3 colour channels.
	int colour_channel = id / image_size;  // 0 - red, 1 - green, 2 - blue.

	B[id] = 255 - A[id];
}


kernel void rgb2grey(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3;  // Each image consists of 3 colour channels.
	int cc = id / image_size;  // colour channel: 0 - red, 1 - green, 2 - blue.

	// Multiplier for RGB values to make greyscale.
	float multiplier[3] = { 0.2126, 0.7152, 0.0722 };
	// Offsets when in each colour channel to get to other channels (RGB).
	int offsets[3][2] = {	{ 1,  2},  // Red.
							{-1,  1},  // Green.
							{-1, -2}   // Blue.
						};

	B[id] += multiplier[cc] * A[id];
	B[id + offsets[cc][0]*image_size] += multiplier[cc] * A[id];
	B[id + offsets[cc][1]*image_size] += multiplier[cc] * A[id];
}


// Simple ND identity kernel.
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0);		// Image width in pixels.
	int height = get_global_size(1);	// Image height in pixels.
	int image_size = width*height;		// Image size in pixels.
	int channels = get_global_size(2);  // Number of colour channels: 3 for RGB.

	int x = get_global_id(0);  // Current x coord.
	int y = get_global_id(1);  // Current y coord.
	int c = get_global_id(2);  // Current colour channel.

	int id = x + y*width + c*image_size;  // Global id in 1D space.

	B[id] = A[id];
}

// 2D averaging filter.
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0);		// Image width in pixels.
	int height = get_global_size(1);	// Image height in pixels.
	int image_size = width*height;		// Image size in pixels.
	int channels = get_global_size(2);  // Number of colour channels: 3 for RGB.

	int x = get_global_id(0);  // Current x coord.
	int y = get_global_id(1);  // Current y coord.
	int c = get_global_id(2);  // Current colour channel.

	int id = x + y*width + c*image_size;  // Global id in 1D space.

	uint result = 0;

	// Simple boundary handling - just copy the original pixel.
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		int r = 2;  // Range.
		for (int i = (x-r); i <= (x+r); i++) {
			for (int j = (y-r); j <= (y+r); j++) {
				result += A[i + j*width + c*image_size];
			}
		}
		result /= (2*r + 1) * (2*r + 1);
	}

	B[id] = (uchar)result;
}

// 2D 3x3 convolution kernel.
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0);		// Image width in pixels.
	int height = get_global_size(1);	// Image height in pixels.
	int image_size = width*height;		// Image size in pixels.
	int channels = get_global_size(2);	// Number of colour channels: 3 for RGB.

	int x = get_global_id(0);  // Current x coord.
	int y = get_global_id(1);  // Current y coord.
	int c = get_global_id(2);  // Current colour channel

	int id = x + y*width + c*image_size;  // Global id in 1D space.

	float result = 0;

	// Simple boundary handling - just copy the original pixel.
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}