/*******************************************************************************
 * Divides the pixel data based on the rectangle drawn on the image
 * Input: Image name. Image must be in a directory named "input/"
 *        Rectangle co-ordinates must be in directory named "bounding boxes/"
 * 		  File contianing rectangle co-ordinates must have same name as image with extension .txt
 * Output:Pixels inside the rectangle are put in file "input/data/fgd"
 *  	  Pixels outside the rectangle are put in fiel "input/data/bgd" 
 *******************************************************************************/

//Author: Mahesh Jagtap

#include "opencv2/opencv.hpp"
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char*argv[]){

	Mat image;
	string input_image = "input image/" + string(argv[1]);
	image = imread(input_image);
	if(image.empty())
		cout << "Could not read image file" << endl;

	string rect_file = "bounding boxes/" + string(argv[1]);
	rect_file.replace(rect_file.length()-3, 3, "txt");
	ifstream rect(rect_file.c_str());
	if(!rect){
		cout << "Cannot open rect file" << endl;
		return -1;
	}
	double x1,y1,x2,y2;
	rect >> x1 >> y1 >> x2 >> y2;

	ofstream fgd_file("input/data/fgd");
	ofstream bgd_file("input/data/bgd");
	int nfgd=0, nbgd=0;
	for(int i=0; i<image.rows; i++){
		for(int j=0; j<image.cols; j++){
			if( i>=x1 && i<=x2 && j>=y1 && j<=y2){
				fgd_file << (int)image.at<Vec3b>(i,j)[0] << " ";
				fgd_file << (int)image.at<Vec3b>(i,j)[1] << " ";
				fgd_file << (int)image.at<Vec3b>(i,j)[2] << endl;
				nfgd++;
			}
			else{
				bgd_file << (int)image.at<Vec3b>(i,j)[0] << " ";
				bgd_file << (int)image.at<Vec3b>(i,j)[1] << " ";
				bgd_file << (int)image.at<Vec3b>(i,j)[2] << endl;
				nbgd++;
			}
		}
	}
	cout << "Data file created.." << endl;
}
