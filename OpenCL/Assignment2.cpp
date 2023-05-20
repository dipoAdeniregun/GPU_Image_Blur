#ifdef __APPLE__
#include "./cl.hpp"             //C++ bindings
#include <OpenCL/opencl.h>		//C header
#else
#include <CL/cl.h>
#include <CL/cl.hpp>
#endif

#include <iostream>
#include <vector>
#include <limits.h>
#include <fstream>
#include <cassert>
#include <math.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp> //changes may be required to opencv includes
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>

void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop)
{
    int num_cols = in.cols;
    int dummy = 0;

    for(int irow = rowstart; irow < rowstop; irow++)
    {
        for(int icol = 0; icol < num_cols; icol++)
        {
            if(level != 0){
                //start a new sum for each pixel
                int pixel_sum[3] = {0, 0 ,0};
                int total_pixels = 0;

                for(int blur_row = irow-level; blur_row < irow+level; blur_row++)
                {
                    for(int blur_col = icol-level; blur_col < icol+level; blur_col++)
                    {
                        //get the total sum surrounding the pixel for each channel
                        //TODO = 
                        for(int i_channel = 0; i_channel < 3; i_channel++){
                            if(blur_row >= 0 && blur_row < rowstop && blur_col >= 0 && blur_col < num_cols ){
                                pixel_sum[i_channel] += in.at<cv::Vec3b>(blur_row, blur_col).val[i_channel];
                                
                            }
                                
                        }
                        total_pixels++;
                        
                    }
                    
                }

                //get the average for the pixel on each channel
                for(int i_channel = 0; i_channel < 3; i_channel++)
                {
                            pixel_sum[i_channel] = (int) (( pixel_sum[i_channel]) / ((double)total_pixels));
                            
                            //if (pixel_sum[i_channel] > 256 || pixel_sum[i_channel] < 0)
                                //std::cout<< pixel_sum[i_channel] ;
                            //assert((pixel_sum[i_channel] <= 256) && (pixel_sum[i_channel] >= 0));
                            out.at<cv::Vec3b>(irow, icol).val[i_channel] = pixel_sum[i_channel];
                }
             
            }
            //out.at<cv::Vec3b>(irow, icol) = in.at<cv::Vec3b>(irow, icol);
        }

            /* 
            this line just copies the pixel over, you need to calculate the new blurred pixel value above
             */
       
        
    }
}

//----------------------------------------------------
// PLATFORM SETUP (C++)
// Return a list of available OpenCL Platforms.
//      Uses C++ Bindings
//----------------------------------------------------
cl_int getOpenCLPlatformsCPP(std::vector<cl::Platform>& platforms, bool verbose)
{
    cl_int err;
	
    //-----------------------------------
    // Get Platforms and Handle Errors
    //-----------------------------------
    platforms.resize(0);
	err = cl::Platform::get(&platforms);
	
    if (err != CL_SUCCESS)
    {
        std::cerr << "Unable to get platforms. Error code = " << err << std::endl;
        return err;
    }

    //-----------------------------------
    // Get and Output Platform Information
    //-----------------------------------
    if (verbose == true)
    {
        for (unsigned int iplat = 0; iplat < platforms.size(); iplat++)
        {
            std::string platform_vendor = platforms[iplat].getInfo<CL_PLATFORM_VENDOR>(&err); //error can be obtained
            std::string platform_version = platforms[iplat].getInfo<CL_PLATFORM_VERSION>(); //or error can be omitted
            std::string platform_name = platforms[iplat].getInfo<CL_PLATFORM_NAME>();
            std::string platform_extensions = platforms[iplat].getInfo<CL_PLATFORM_EXTENSIONS>();
		
            std::cout << "Information for Platform " << iplat << ": " << std::endl;
            std::cout << "\tVendor		: " << platform_vendor << std::endl;
            std::cout << "\tVersion		: " << platform_version << std::endl;
            std::cout << "\tName		: " << platform_name << std::endl;
            std::cout << "\tExtensions	: " << platform_extensions << std::endl;
        }
        std::cout << "\n\n";
    }
    return err;
}

//----------------------------------------------------
// Struct for Storing Device Info
// A lot more could be added, just some basics
//----------------------------------------------------
typedef struct
{
	std::string device_name;
	std::string device_vendor;
	std::string device_version;
	cl_device_type device_type;
 	cl_bool device_available;
 	cl_bool device_compiler_available;
	std::string device_extensions;
	cl_ulong device_global_memory_size; //bytes
	cl_ulong device_local_memory_size; //bytes
	cl_ulong device_max_mem_alloc_size; //bytes
	cl_ulong device_global_mem_cache_size; //bytes
	cl_uint device_global_mem_cacheline_size; //bytes
	cl_uint device_max_compute_units;
	cl_uint device_max_clock_frequency; //MHz
	size_t device_max_work_group_size;
	cl_uint device_max_work_item_dimensions;
    size_t device_max_work_item_sizes[3];   //assuming max dimensions is three
    
}
device_info_t;

//----------------------------------------------------
// Function for Getting Device Info
//----------------------------------------------------
void getDeviceInfo(cl::Device &device, device_info_t& info)
{
	info.device_name = device.getInfo<CL_DEVICE_NAME>();
	info.device_type = device.getInfo<CL_DEVICE_TYPE>();
	info.device_vendor = device.getInfo<CL_DEVICE_VENDOR>();
	info.device_version = device.getInfo<CL_DEVICE_VERSION>();
	info.device_available = device.getInfo<CL_DEVICE_AVAILABLE>();
	info.device_compiler_available = device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
	info.device_extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
	info.device_global_memory_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	info.device_local_memory_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
	info.device_max_mem_alloc_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
	info.device_global_mem_cache_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
	info.device_global_mem_cacheline_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
	info.device_max_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	info.device_max_clock_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
	info.device_max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	info.device_max_work_item_dimensions = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    
    for (int idim = 0; idim < 3; idim++)
    {
        info.device_max_work_item_sizes[idim] = -1;
    }
    
    for (int idim = 0; idim < info.device_max_work_item_dimensions; idim++)
    {
        info.device_max_work_item_sizes[idim] = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[idim];
    }
}

//----------------------------------------------------
// Function for Printing Device Info
//----------------------------------------------------
void printDeviceInfo(const device_info_t& info)
{
	std::cout << "\nDevice Information " << std::endl;
	std::cout << "\tName                    : " << info.device_name << "\n";
	std::cout << "\tType                    : ";
	if (info.device_type == CL_DEVICE_TYPE_CPU) std::cout << "CPU\n";
	else if (info.device_type == CL_DEVICE_TYPE_GPU) std::cout << "GPU\n";
	else if (info.device_type == CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR\n";
	else if (info.device_type == CL_DEVICE_TYPE_DEFAULT) std::cout << "DEFAULT\n";
	else std::cout << "(ERROR!)\n";
	std::cout << "\tVendor                  : " << info.device_vendor << "\n";
	std::cout << "\tVersion                 : " << info.device_version << "\n";
	std::cout << "\tAvailable               : " << info.device_available << "\n";
	std::cout << "\tCompiler Available      : " << info.device_compiler_available << "\n";
	std::cout << "\tGlobal Memory Size      : " << info.device_global_memory_size/(1024*1024*1024.0) << " GB\n";
	std::cout << "\tLocal Memory Size       : " << info.device_global_memory_size/(1024*1024*1024.0) << " GB\n";
	std::cout << "\tMaximum Memory Alloc    : " << info.device_max_mem_alloc_size/(1024*1024.0) << " MB\n";
	std::cout << "\tGlobal Memory Cache     : " << info.device_global_mem_cache_size << " Bytes\n";
	std::cout << "\tGlobal Memory CacheLine : " << info.device_global_mem_cacheline_size << " Bytes\n";
	std::cout << "\tMax Compute Units       : " << info.device_max_compute_units << "\n";
	std::cout << "\tMax Clock Frequency     : " << info.device_max_clock_frequency << " MHz\n";
	std::cout << "\tMax Work Group Size     : " << info.device_max_work_group_size << "\n";
	std::cout << "\tMax Work Group Dim      : " << info.device_max_work_item_dimensions << "\n";
    
    int maximum_work_item_product = 1;
    for (int idim = 0; idim < info.device_max_work_item_dimensions; idim++)
    {
        std::cout << "\tMax Work Item Size - Dimension " << idim << " : " << info.device_max_work_item_sizes[idim] << "\n";
        maximum_work_item_product *= info.device_max_work_item_sizes[idim];
    }
    std::cout << "\tMaximum Number of Work Items Globally   : " << maximum_work_item_product << "\n";
}



//----------------------------------------------------
// DEVICE SETUP
// Return a list of Devices from a Platform
//----------------------------------------------------

cl_int getOpenCLDevices(const cl::Platform& platform, std::vector<cl::Device>& devices, cl_device_type device_type, bool verbose)
{
    if (device_type != CL_DEVICE_TYPE_ALL && device_type != CL_DEVICE_TYPE_CPU && device_type != CL_DEVICE_TYPE_GPU)
    {
        std::cerr << "Unknown Device Type!" << std::endl;
        std::cerr << "Assuming ALL Devices" << std::endl;
        device_type = CL_DEVICE_TYPE_ALL;
    }

    cl_int err = platform.getDevices(device_type, &devices);

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Occurred Calling Platform::getDevices(). Error code = " << err << std::endl;
        return err;
    }

    if (verbose == true)
    {
        for (unsigned int idevice = 0; idevice < devices.size(); idevice++)
        {
            device_info_t device_info;
            getDeviceInfo(devices[idevice], device_info);
            printDeviceInfo(device_info);
        }
    }
    return err;
}



int main(int argc, char** argv)
{
	cl_int err;

    //----------------------------------------------------
	// PLATFORM SETUP
    //----------------------------------------------------
    
    std::cout << "------------------------------------------\n";
    std::cout << " Getting Platforms\n";
    std::cout << "------------------------------------------\n";
    
    std::vector<cl::Platform> platforms;
    err = getOpenCLPlatformsCPP(platforms,false);   //no output
    
    std::cout << "------------------------------------------\n";
    std::cout << " Getting Platform Devices\n";
    std::cout << "------------------------------------------\n";

    //----------------------------------------------------
	// DEVICES SETUP
    //----------------------------------------------------

    std::vector<cl::Device> gpu_devices;
    err = getOpenCLDevices(platforms[0], gpu_devices, CL_DEVICE_TYPE_GPU, false);
    
    //----------------------------------------------------
	// CONTEXT SETUP
    //----------------------------------------------------

    std::cout << "-------------------------\n";
    std::cout << " Creating GPU Context\n";
    std::cout << "-------------------------\n";

    //----------------------
	// Create Context
    //----------------------
	
    cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
	cl::Context context(gpu_devices, context_properties, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Creating Context. Error Code: " << err << std::endl;
        return -1;
    }
	
    //----------------------------------------------------
	// COMMAND QUEUE SETUP
    //----------------------------------------------------
    
    std::cout << "\n---------------------------------------------\n";
    std::cout << " Creating Command Queue \n";
    std::cout << "---------------------------------------------\n";
    
    //----------------------
	// Create Command Queue
    //----------------------
    
    cl::CommandQueue gpu_command_queue(context, gpu_devices[0],0,&err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Creating Command Queue. Error Code: " << err << std::endl;
        return -1;
    }

    //----------------------------------------------------
	// PROGRAM SETUP
    //----------------------------------------------------
    
    std::cout << "\n---------------------------------------------\n";
	std::cout << "Program Info: " << std::endl;
	std::cout << "---------------------------------------------\n";
    
    //----------------------------
	// Read the kernel file into a string
    //----------------------------
    
    std::ifstream in_from_source("./imageblur.cl");
    std::string kernel_code(std::istreambuf_iterator<char>(in_from_source), (std::istreambuf_iterator<char>()));

    
    //----------------------------
	// Format the kernel string properly
    // for program constructor
    //----------------------------
    
    cl::Program::Sources kernel_source(1,std::make_pair(kernel_code.c_str(), kernel_code.length() + 1));
    
    //----------------------------
	// Create a Program in this context for that source
    //----------------------------
	cl::Program program(context, kernel_source, &err);

    if (err != CL_SUCCESS)
	{
		std::cerr << "Error Creating Program. Error Code: " << err << std::endl;
		return -1;
	}

    // std::cout << "The source in the program is: \n";
    // std::cout << program.getInfo<CL_PROGRAM_SOURCE>();
    // std::cout << "\n\n";
    
    
    //----------------------------
	// Build the functions in the program for devices
    //----------------------------
    err = program.build(gpu_devices,NULL,NULL,NULL);
    
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error Building Program. Error Code: " << err << std::endl;
        std::string name     = gpu_devices[0].getInfo<CL_DEVICE_NAME>();
        std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu_devices[0]);
        std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Program Built Successfully." << std::endl;
    }
    
    std::vector<size_t> program_binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    assert(program_binary_sizes.size() == gpu_devices.size());
    std::cout << "Binaries size = " << program_binary_sizes[0] << " bytes" <<  std::endl;
    
    
    //----------------------------------------------------
	// KERNEL SETUP
    //----------------------------------------------------
    
    std::cout << "\n---------------------------------------------\n";
	std::cout << "Creating Kernel: " << std::endl;
	std::cout << "---------------------------------------------\n";
    
    //----------------------------
	// Create Kernel
    //----------------------------
    std::string kernel_function_name = "imageblur";
    cl::Kernel kernel(program,kernel_function_name.c_str(), &err);
    
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Creating Kernel Object. Error Code: " << err << std::endl;
    }
    else
    {
        std::cout << "Kernel Created Successfully" << std::endl;
    }
    std::cout << "Kernel Function Name = " << kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() << std::endl;
    std::cout << "Kernel Number of Arguments = " << kernel.getInfo<CL_KERNEL_NUM_ARGS>() << std::endl;
    
    
    
    //----------------------------
	// Kernel Argument Allocation
    //----------------------------
    
    if(argc < 5){
        std::cout << "Must provide level and valid image name as arguments!\n";
        return 1;
    }

    int level = atoi(argv[2]);
    int block_size_x = atoi(argv[3]);
    int block_size_y = atoi(argv[4]);
    cv::Mat picture, newImg; 
    picture = cv::imread(argv[1],1);
    //image = image.reshape(3,1);
    newImg = picture.clone();
    std::vector<unsigned char> image(picture.rows*picture.cols*3);
    int count = 0;
    for(int i = 0; i < picture.rows; i++){
        for(int j = 0; j < picture.cols; j++){
            for(int k = 0; k < 3; k++){
                image[count++] = picture.at<cv::Vec3b>(i,j).val[k];
            }
        }
    }
        
    //---------------------------
    // Start GPU Timer
    //---------------------------
    
    gpu_command_queue.finish();
    
    std::cout << "Starting GPU Timing" << std::endl;
    timeval gpu_t_start;
    gettimeofday(&gpu_t_start,0x0);
    
    

    
    
    //---------------------------
    // Create Buffers
    //---------------------------
    
    cl::Buffer cl_image_in(context, CL_MEM_READ_ONLY, 3*image.size()*sizeof(char), NULL, &err); assert(err == CL_SUCCESS);
    //cl::Buffer cl_b(context, CL_MEM_READ_ONLY, b.size()*sizeof(float), NULL, &err); assert(err == CL_SUCCESS);
    cl::Buffer cl_image_out(context, CL_MEM_WRITE_ONLY, 3*image.size()*sizeof(char), NULL, &err); assert(err == CL_SUCCESS);
    
    //---------------------------
    // Put Buffers in Write Queue
    // - this passes to GPU
    //---------------------------
    
    err = gpu_command_queue.enqueueWriteBuffer(cl_image_in, CL_TRUE, 0, 3*image.size()*sizeof(char), &image[0], NULL, NULL); assert(err == CL_SUCCESS);
    //err = gpu_command_queue.enqueueWriteBuffer(cl_b, CL_TRUE, 0, b.size()*sizeof(float), &b[0], NULL, NULL); assert(err == CL_SUCCESS);
    err = gpu_command_queue.enqueueWriteBuffer(cl_image_out, CL_TRUE, 0, 3*image.size()*sizeof(char), &image[0], NULL, NULL); assert(err == CL_SUCCESS);
    

    //---------------------------
    // Set Kernel Arguments
    //---------------------------
    
    err = kernel.setArg(0, cl_image_in);
    err = kernel.setArg(1, cl_image_out);
    err = kernel.setArg(2, picture.rows);
    err = kernel.setArg(3, picture.cols);
    err = kernel.setArg(4, level);
    err = kernel.setArg(5, 0);

    //----------------------------------------------------
	// KERNEL EXECUTION
    //----------------------------------------------------
    
    std::cout << "\n---------------------------------------------\n";
	std::cout << "Executing Kernel: " << std::endl;
	std::cout << "---------------------------------------------\n";
    
    //---------------------------
    // Execute Kernel
    //---------------------------
    
    err = gpu_command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(picture.rows,picture.cols), cl::NullRange, NULL, NULL);
    
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Enqueueing NDRange Kernel. Error Code: " << err << std::endl;
    }
    else
    {
        std::cout << "NDRange Kernel Enqueued Successfully" << std::endl;
    }

    err = kernel.setArg(5, 1);

    err = gpu_command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(picture.rows,picture.cols), cl::NullRange, NULL, NULL);
    
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Enqueueing NDRange Kernel. Error Code: " << err << std::endl;
    }
    else
    {
        std::cout << "NDRange Kernel Enqueued Successfully" << std::endl;
    }

    err = kernel.setArg(5, 2);

    err = gpu_command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(picture.rows,picture.cols), cl::NullRange, NULL, NULL);
    
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Enqueueing NDRange Kernel. Error Code: " << err << std::endl;
    }
    else
    {
        std::cout << "NDRange Kernel Enqueued Successfully" << std::endl;
    }

    //---------------------------
    // Read Result
    //---------------------------

    err = gpu_command_queue.enqueueReadBuffer(cl_image_out, CL_TRUE, 0, image.size()*sizeof(char), &newImg.at<cv::Vec3b>(0), NULL, NULL);

    //---------------------------
    // Stop GPU Timer
    //---------------------------
    
    gpu_command_queue.finish();
    
    timeval gpu_t_stop;
    gettimeofday(&gpu_t_stop,0x0);
    
    double gpu_time = (gpu_t_stop.tv_sec - gpu_t_start.tv_sec) + ((gpu_t_stop.tv_usec - gpu_t_start.tv_usec)/1000000.0);
    
    std::cout << "GPU Computation Complete" << std::endl;
    
    cv::Mat serial_blur = picture.clone();

    double cpu_t_start = (double)clock()/(double)CLOCKS_PER_SEC;
    imageBlur(picture, serial_blur, level, 0, picture.rows);
    double cpu_t_end = (double)clock()/(double)CLOCKS_PER_SEC;

    imwrite("Serial_Blur_level3.jpg", serial_blur);
    
    
    
    //----------------------------------------------
    // Display Timing Results and Images
    //----------------------------------------------
    
    std::cout << "GPU Time = " << gpu_time << "s, CPU Time = "<< cpu_t_end - cpu_t_start << "s" << std::endl;
    std::cout << "Speedup = " << (cpu_t_end - cpu_t_start)/gpu_time << std::endl;
    cv::imshow("blurred Image", newImg);
    cv::imwrite("OpenCL_Blur_level3.jpg", newImg);
    cv::waitKey(9000);
}


