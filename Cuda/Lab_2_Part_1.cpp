#include <iostream>
#include <opencv2/core/core.hpp>            //changes may be required to opencv includes
#include <opencv2/highgui/highgui.hpp>
#include <cassert>

using namespace std;

void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop);

int main(int argc, char** argv)
{
    cv::Mat image;
    image = cv::imread("./space2.jpg",1);   // Read the file

    if(! image.data )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    int level = 3;

    cv::Mat blurred_image = image.clone();

    imageBlur(image, blurred_image, level, 0, image.rows);

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow("Image", blurred_image);                   // Show our image inside it.
    cv::imwrite("serial_blur.jpg", blurred_image);

    cv::waitKey(10000);

    return 0;
}

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