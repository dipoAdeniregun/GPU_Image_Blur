#include <iostream>
#include <vector>
#include <cassert>
#include <cuda.h>	//required for CUDA
#include <opencv2/core/core.hpp> //changes may be required to opencv includes
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>

#define CHANNELS 3;

void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop)
{
    int num_cols = in.cols;
    //int dummy = 0;

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

//---------------------------------------
//CUDA C++ image blur kernel function
//---------------------------------------
__global__ void imageBlur(unsigned char *in, unsigned char *out, int rows, int cols, int level, int channel)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int channel = blockIdx.z * blockDim.z + threadIdx.z;

    //loop over rows to be used in blur
    int pixels = 0;
    int totalChannel = 0;

    if(col < cols && row < rows){
        for(int blurRow = (row - level); blurRow < (row + level); blurRow++){
        //loop over columns to be used in blur
            for(int blurCol = (col - level); blurCol < (col + level); blurCol++){
                //loop over channels
                if(blurRow >= 0 && blurRow < rows && blurCol >= 0 && blurCol < cols ){

                    totalChannel += in[blurRow*cols*3 + blurCol*3+channel];
                    pixels++;
                }
            
            }
        }
    
        out[row*cols*3 + col*3+channel] = totalChannel/pixels;
        //ensure valid pixel value
        assert( out[row*cols*3 + col*3+channel] < 256);

    }
        
 
    
}

int main (int argc, char** argv){
    cudaError_t err;

    cudaDeviceProp device_properties;
	cudaGetDeviceProperties	(&device_properties, 0);

    if(argc < 5){
        std::cout << "Must provide level and valid image name as arguments!\n";
        return 1;
    }

    int level = atoi(argv[2]);
    int block_size_x = atoi(argv[3]);
    int block_size_y = atoi(argv[4]);
    cv::Mat picture, newImg; 
    picture = cv::imread(argv[1],1);
    cv::Mat serial_blur = picture.clone();
    double cpu_t_start = (double)clock()/(double)CLOCKS_PER_SEC;
    imageBlur(picture, serial_blur, level, 0, picture.rows);
    double cpu_t_end = (double)clock()/(double)CLOCKS_PER_SEC;

    /****************************************************************************
    * segfaults when we attempt to display or write the image from the serial blur function. 
    * Image appears to be correct when displayed
    ****************************************************************************/
    // cv::imshow("CPU_Blur",serial_blur);
    // cv::waitKey(1000);
    
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

    //---------------------------------------
    // Setup Profiling
    //---------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //---------------------------------------
	// Create GPU (Device) Buffers
	//---------------------------------------
    unsigned char *original;
    unsigned char *blurred;
    int size = image.size()*sizeof(char);

    err = cudaMalloc((void**)&original, 3*size); assert (err == cudaSuccess);
    err = cudaMalloc((void**)&blurred, 3*size); assert (err == cudaSuccess);

    //---------------------------------------
	// Copy Memory To Device
	// No need to copy c, memory already
    // allocated above
	//---------------------------------------

    err = cudaMemcpy(original, &image[0], 3*size, cudaMemcpyHostToDevice); assert(err == cudaSuccess);
    
    //---------------------------------------
	// Setup Execution Configuration
	//---------------------------------------
    
    dim3 block_size(block_size_x,block_size_y);

    int gridy = (int)ceil((double)picture.rows/(double)block_size_y);
    int gridx = (int)ceil((double)picture.cols/(double)block_size_x);

    dim3 grid_size(gridx , gridy);

    if (block_size.x*block_size.y > device_properties.maxThreadsPerBlock)
	{
		std::cerr << "Block Size of " << block_size.x << " x " << block_size.y << " is too big. " << std::endl;
		std::cerr << "Maximum threads per block = " << device_properties.maxThreadsPerBlock << std::endl;
		return -1;
	}
	else if (block_size.x > device_properties.maxThreadsDim[0] || block_size.y > device_properties.maxThreadsDim[1])
	{
		std::cerr << "Block Size of " << block_size.x << " x " << block_size.y << " is too big. " << std::endl;
		std::cerr << "Maximum threads for dimension 0 = " << device_properties.maxThreadsDim[0] << std::endl;
		std::cerr << "Maximum threads for dimension 1 = " << device_properties.maxThreadsDim[1] << std::endl;
		return -1;
	}
	else if (grid_size.x > device_properties.maxGridSize[0] || grid_size.y > device_properties.maxGridSize[1])
	{
		std::cerr << "Grid Size of " << grid_size.x << " x " << grid_size.y << " is too big. " << std::endl;
		std::cerr << "Maximum grid dimension 0 = " << device_properties.maxGridSize[0] << std::endl;
		std::cerr << "Maximum grid dimension 1 = " << device_properties.maxGridSize[1] << std::endl;		
		return -1;
	}

    //---------------------------------------
	// Call Kernel
	//---------------------------------------
    imageBlur<<<grid_size, block_size>>>(original, blurred, picture.rows, picture.cols, level, 0);
    imageBlur<<<grid_size, block_size>>>(original, blurred, picture.rows, picture.cols, level, 1);
    imageBlur<<<grid_size, block_size>>>(original, blurred, picture.rows, picture.cols, level, 2);

    //---------------------------------------
	// Obtain Result
	//---------------------------------------
    
    err = cudaMemcpy(&newImg.at<cv::Vec3b>(0),blurred, 3*size, cudaMemcpyDeviceToHost); assert(err == cudaSuccess);    

    //---------------------------------------
	// Stop Profiling
	//---------------------------------------
	
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);   //time in milliseconds
    
    gpu_time /= 1000.0;
    std::cout << "Done GPU Computations in " << gpu_time << " seconds" << std::endl; std::cout.flush();
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(original);
    cudaFree(blurred);
    //----------------------------------------------------
	// CPU COMPUTATION
    //----------------------------------------------------

    
    

    std::cout << "---------------------------------------------------\n";
    std::cout << "Results\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "For block size (" << block_size_x << "x" << block_size_y << ")" << " and grid size (" << gridx << "x" << gridy << ")\n";
    std::cout << "Gpu time = " << gpu_time << "s. Cpu time = " << cpu_t_end - cpu_t_start << "s.\n";
    std::cout << "Speedup = " << (cpu_t_end - cpu_t_start)/gpu_time << std::endl;  

    

    cv::imshow("blurred Image", newImg);
    cv::waitKey(500);
    cv::imwrite("CudaBlur.jpg", newImg);
    cv::imwrite("CPUBlur.jpg", serial_blur);
    cv::waitKey(9000);
}
