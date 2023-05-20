__kernel void imageblur(__global unsigned char* in, __global unsigned char* out, int rows, int cols, int level, int channel)
{
    int col = get_global_id(1);
    int row = get_global_id(0);

    //int channel = blockIdx.z * blockDim.z + threadIdx.z;

    //loop over rows to be used in blur
    int pixels = 0;
    int totalChannel = 0;
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
    //assert( out[row*cols*3 + col*3+channel] < 256);
    
}
