/*Calculate maximum likelihood
 *Divide data according to the labels assigned */

/*Author: Mahesh Jagtap */

#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"
#include <fstream>

using namespace std;
using namespace cv;

void getData(vector<Vec3d>& vSamples, string fgdbgd){
	string file("input/data/");
	file += fgdbgd;
	
	ifstream data_file(file.c_str());
	if(!data_file){
		cout << "Could not open data file" << endl;
		return;
	}
	string line;
	while( getline(data_file,line) ){
		stringstream s(line);
		Vec3d v;
		s >> v[0] >> v[1] >> v[2];
		vSamples.push_back(v);
	}
	data_file.close();
}

void likelihood(Mat& samples, string fgdbgd, string ks){
	stringstream s;
	s << ks;
	int k;
	s >> k;

	EM model(k,EM::COV_MAT_GENERIC);

	Mat meansInitial = Mat::zeros(k,3,CV_64FC1);
	Mat weightsInitial = Mat::zeros(1,k,CV_64FC1);
	vector<Mat> covsInitial;

	Mat log_likelihoods, labels;
	
	model.train(samples,log_likelihoods, labels);
	
	//Divide samples according to the components
	vector<Vec3b> components[k];
	for(int i=0; i<samples.rows; i++){
		Vec3b v;
		double* rowi = samples.ptr<double>(i);
		v[0] = (int)rowi[0];
		v[1] = (int)rowi[1];
		v[2] = (int)rowi[2];
		components[labels.at<int>(i,0)].push_back(v);
	}
	for(int j=0; j<k; j++){
		stringstream s;
		s << j;
		string curK;
		s >> curK;
		string file = "input/components/" + curK; 
		ofstream out(file.c_str());

		for(int i=0; i<components[j].size(); i++){
			out << (int)components[j][i][0] << " ";
			out << (int)components[j][i][1] << " ";
			out << (int)components[j][i][2] << endl;
		}
	}
}

//two arguments
//argv[1] = fgd or bgd
//argv[2] = number of components
int main(int argc, char* argv[]){

	vector<Vec3d> vSamples;
	getData(vSamples, argv[1]);
	
	Mat samples((int)vSamples.size(),3,CV_64FC1,&vSamples[0][0]);

	likelihood(samples, argv[1], argv[2]);	
}
