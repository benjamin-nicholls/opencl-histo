#include "Utils.h"
#include "CImg.h"
#include <math.h>


void const print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
	std::cerr << "  -b : specify number of bins (default: 256)" << std::endl; 
}


void const printIntVector(const std::vector<int> v, const int MAX_VECTOR_PRINT_SIZE = 500) {
	if (v.size() < MAX_VECTOR_PRINT_SIZE) {
		std::cout << v << std::endl;
	} else {
		std::cout << "(Vector too large to print)" << std::endl;
	}
}


int main(int argc, char** argv) {
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int n_bins = 256;
	bool colour_image = false;
	const int MAX_VECTOR_PRINT_SIZE = 500;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { n_bins = atoi(argv[++i]); }  // Added for number of bins.
	}

	cimg_library::cimg::exception_mode(0);

	// Detect any potential exceptions.
	try {
		cimg_library::CImg<unsigned char> image_input(image_filename.c_str());
		cimg_library::CImg<unsigned short> image_input_16(image_filename.c_str());
		cimg_library::CImgDisplay disp_input;
		bool is_16bit = image_input_16.max() > 255;

		if (is_16bit) {
			// Convert 16-bit image to 8-bit and scale to [0,255] range.
			// image_input /= (1 << 8);  // remove
			image_input = image_input_16.get_normalize(0,255);
			disp_input.set_title("Input Image (16-bit converted to 8-bit)");
		} else {
			disp_input.set_title("Input Image (8-bit)");
		}
		delete[] image_input_16;
		disp_input.display(image_input);


		// Part 3 - host operations.
		// 3.1 Select computing devices.
		cl::Context context = GetContext(platform_id, device_id);
		// Display the selected device.
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		// Create a queue to which we will push commands for the device.
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		// 3.2 Load & build the device code.
		cl::Program::Sources sources;
		AddSources(sources, "kernels/kernels.cl");
		cl::Program program(context, sources);

		// Build and debug the kernel code.
		try {
			program.build();
		} catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Part 4 - Device Operations.
		typedef int mytype;
		std::vector<mytype> h(n_bins);  // Histogram.
		size_t h_size = h.size() * sizeof(mytype);

		cimg_library::CImg<unsigned char> Cb;
		cimg_library::CImg<unsigned char> Cr;

		// If RGB, convert to YCbCr.
		if (image_input.spectrum() == 3) {
			image_input = image_input.get_RGBtoYCbCr();
			Cb = image_input.get_channel(1);			// Cb
			Cr = image_input.get_channel(2);			// Cr
			image_input = image_input.get_channel(0);   // Y
			colour_image = true;
		} else {
			delete[] Cb;
			delete[] Cr;
		}

		// Device - Buffers.
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		// 4.1 Copy images to device memory.
		cl::Event prof_event_img_transfer;
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &prof_event_img_transfer);
		std::cout << "Transfer time: " << GetFullProfilingInfo(prof_event_img_transfer, ProfilingResolution::PROF_US) << "\n" << std::endl;


		/*
			1. CALCULATE AN INTENSITY HISTOGRAM FROM THE INPUT IMAGE.
		*/
		// Buffer
		cl::Buffer dev_h_output(context, CL_MEM_READ_WRITE, h_size);
		queue.enqueueFillBuffer(dev_h_output, 0, 0, h_size);
		// Kernel
		cl::Kernel kernel_h = cl::Kernel(program, "hist_simple");
		kernel_h.setArg(0, dev_image_input);
		kernel_h.setArg(1, dev_h_output);
		kernel_h.setArg(2, n_bins);
		// Profiling event
		cl::Event prof_event_h;
		// Queue
		queue.enqueueNDRangeKernel(kernel_h, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event_h);
		// Read buffer
		queue.enqueueReadBuffer(dev_h_output, CL_TRUE, 0, h_size, &h.data()[0]);
		std::cout << "histogram, bins=" << n_bins << ", h=" << std::endl;
		printIntVector(h, MAX_VECTOR_PRINT_SIZE);
		std::cout << "Kernel execution time [ns]:" <<
			prof_event_h.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_h.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Execution time: " << GetFullProfilingInfo(prof_event_h, ProfilingResolution::PROF_US) << "\n" << std::endl;


		/*
			2a. CALCULATE A CUMULATIVE HISTOGRAM.
		*/
		std::vector<mytype> hc(n_bins);  // Histogram cumulative.
		// Buffer
		cl::Buffer dev_hc_output(context, CL_MEM_READ_WRITE, h_size);
		queue.enqueueFillBuffer(dev_hc_output, 0, 0, h_size);
		// Kernel
		cl::Kernel kernel_hc = cl::Kernel(program, "hist_cumulative");
		kernel_hc.setArg(0, dev_h_output);
		kernel_hc.setArg(1, dev_hc_output);
		// Profling event
		cl::Event prof_event_hc;
		// Queue
		queue.enqueueNDRangeKernel(kernel_hc, cl::NullRange, cl::NDRange(h_size), cl::NullRange, NULL, &prof_event_hc);
		// Read buffer
		queue.enqueueReadBuffer(dev_hc_output, CL_TRUE, 0, h_size, &hc.data()[0]);

		std::cout << "cumulative histogram, hc=" << std::endl;
		printIntVector(hc, MAX_VECTOR_PRINT_SIZE);
		std::cout << "Kernel execution time [ns]:" <<
			prof_event_hc.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_hc.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Execution time: " << GetFullProfilingInfo(prof_event_hc, ProfilingResolution::PROF_US) << "\n" << std::endl;


		/*
			2b. BLELLOCH SCAN, for comparison, unused otherwise.
		*/
		std::vector<mytype> hcb(n_bins);  // Histogram cumulative.
		// Buffer
		cl::Buffer dev_hc_b_output(context, CL_MEM_READ_WRITE, h_size);
		//queue.enqueueFillBuffer(dev_hc_b_output, 0, 0, h_size);
		queue.enqueueCopyBuffer(dev_h_output, dev_hc_b_output, 0, 0, h_size);
		// Kernel
		cl::Kernel kernel_hc_b = cl::Kernel(program, "scan_bl");
		kernel_hc_b.setArg(0, dev_hc_b_output);
		// Profiling event
		cl::Event prof_event_hc_b;
		// Queue
		queue.enqueueNDRangeKernel(kernel_hc_b, cl::NullRange, cl::NDRange(h_size), cl::NullRange, NULL, &prof_event_hc_b);
		// Read buffer
		queue.enqueueReadBuffer(dev_hc_b_output, CL_TRUE, 0, h_size, &hcb.data()[0]);

		std::cout << "cumulative histogram, blelloch, hc=" << std::endl;
		printIntVector(hcb, MAX_VECTOR_PRINT_SIZE);
		std::cout << "Kernel execution time [ns]:" <<
			prof_event_hc_b.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_hc_b.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Execution time: " << GetFullProfilingInfo(prof_event_hc_b, ProfilingResolution::PROF_US) << "\n" << std::endl;

		//for (int a = 0; a < hc.size(); a++) {
			//std::cout << a << ": " << hc[a] << "==" << hcb[a] << "=" << (hc[a] == hcb[a]) << std::endl;
		//}
		

		/*
			2c. HILLIS-STEELE SCAN, for comparison, unused otherwise.
		*/
		std::vector<mytype> hchs(n_bins);  // Histogram cumulative.
		// Buffer
		cl::Buffer dev_hc_hs_output(context, CL_MEM_READ_WRITE, h_size);
		queue.enqueueCopyBuffer(dev_h_output, dev_hc_hs_output, 0, 0, h_size);
		cl::Buffer dev_hc_hs_extra(context, CL_MEM_READ_WRITE, h_size);
		queue.enqueueFillBuffer(dev_hc_hs_extra, 0, 0, h_size);
		// Kernel
		cl::Kernel kernel_hc_hs = cl::Kernel(program, "scan_hs");
		kernel_hc_hs.setArg(0, dev_hc_hs_output);
		kernel_hc_hs.setArg(1, dev_hc_hs_extra);
		// Profiling event
		cl::Event prof_event_hc_hs;
		// Queue
		queue.enqueueNDRangeKernel(kernel_hc_hs, cl::NullRange, cl::NDRange(h_size), cl::NullRange, NULL, &prof_event_hc_hs);

		// Read buffer
		queue.enqueueReadBuffer(dev_hc_hs_output, CL_TRUE, 0, h_size, &hchs.data()[0]);

		std::cout << "cumulative histogram, hillis-steel, hc=" << std::endl;
		printIntVector(hchs, MAX_VECTOR_PRINT_SIZE);
		std::cout << "Kernel execution time [ns]:" <<
			prof_event_hc_hs.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_hc_hs.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Execution time: " << GetFullProfilingInfo(prof_event_hc_hs, ProfilingResolution::PROF_US) << "\n" << std::endl;


		/*
			3. NORMALISE AND SCALE THE CUMULATIVE HISTOGRAM.
		*/
		//std::vector<mytype> lut(n_bins);
		std::vector<mytype> lut(256);
		size_t lut_size = lut.size() * sizeof(mytype);  // Added.
		// Buffer
		cl::Buffer dev_lut_output(context, CL_MEM_READ_WRITE, lut_size);
		queue.enqueueFillBuffer(dev_lut_output, 0, 0, lut_size);
		// Kernel
		cl::Kernel kernel_lut = cl::Kernel(program, "normalise");
		kernel_lut.setArg(0, dev_hc_output);
		kernel_lut.setArg(1, dev_lut_output);
		kernel_lut.setArg(2, n_bins);
		// Profiling event
		cl::Event prof_event_lut;
		// Queue 
		std::cout << "lut.size()=" << lut.size() << std::endl;
		queue.enqueueNDRangeKernel(kernel_lut, cl::NullRange, cl::NDRange(lut.size()), cl::NullRange, NULL, &prof_event_lut);
		// Read buffer
		queue.enqueueReadBuffer(dev_lut_output, CL_TRUE, 0, lut_size, &lut.data()[0]);
		std::cout << "look up table, lut=" << std::endl;
		printIntVector(lut, MAX_VECTOR_PRINT_SIZE);
		std::cout << "Kernel execution time [ns]:" <<
			prof_event_lut.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_lut.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Execution time: " << GetFullProfilingInfo(prof_event_lut, ProfilingResolution::PROF_US) << "\n" << std::endl;


		/*
			4. BACK PROJECTION.
		*/
		// Buffer
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		// Kernel
		cl::Kernel kernel_back_projection = cl::Kernel(program, "back_projection");
		kernel_back_projection.setArg(0, dev_image_input);
		kernel_back_projection.setArg(1, dev_lut_output);
		kernel_back_projection.setArg(2, dev_image_output);
		//kernel_back_projection.setArg(3, lut.size());
		// Profiling event
		cl::Event prof_event_bp;
		// Queue
		queue.enqueueNDRangeKernel(kernel_back_projection, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event_bp);


		// 4.3 Copy the result from device to host.
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		// PREPARE OUTPUT, do colour checks/conversions.
		cimg_library::CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		cimg_library::CImg<unsigned char> recombined_image(output_image.width(), output_image.height(), 1, 3);  // Only used if colour_image = true
	
		if (colour_image) {
			// In-built for loop in cimg_library.
			cimg_forXY(recombined_image, x, y) {
				recombined_image(x, y, 0, 0) = output_image(x, y);  // [x,y,z,channel]
				recombined_image(x, y, 0, 1) = Cb(x, y);
				recombined_image(x, y, 0, 2) = Cr(x, y);
			}
			output_image = recombined_image.get_YCbCrtoRGB();
		}

		std::cout << "back projection, <output image>\n" << std::endl;

		// Display correct image.
		//cimg_library::CImgDisplay disp_output;
		//disp_output.display(output_image);
		//disp_output.set_title("output");
		cimg_library::CImgDisplay disp_output(output_image, "output");  // DISPLAY IMAGE.
		std::cout << "Kernel execution time [ns]:" <<
			prof_event_bp.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event_bp.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Execution time: " << GetFullProfilingInfo(prof_event_bp, ProfilingResolution::PROF_US) << "\n" << std::endl;
		
		std::cout << "TOTAL TIME: " <<
			(prof_event_img_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_img_transfer.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) +
			(prof_event_h.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_h.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) +
			(prof_event_hc.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_hc.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) +
			(prof_event_lut.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_lut.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) +
			(prof_event_bp.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_bp.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) << std::endl;


		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	} catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	} catch (cimg_library::CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}

