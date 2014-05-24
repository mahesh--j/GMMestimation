/*******************************************************************************
 * Algorithm to estimate the number of mixtures of the GMM.
 * Implementation of "CLUSTER: An unsuperwised algorithm for modelling Gaussian mixtures"(by C.A. Bouman)
 * Changes made in the above algorithm:
 	- Uses Mixture-MDL criteria for calculating penalty term. Explained in "On fitting mixture models"(by MAT Figueiredo)    
 *******************************************************************************/

#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

using namespace std;
using namespace cv;

//Merge components c1 and c2
void mergeComponents(const Mat& means, const vector<Mat>& covs, const Mat& weights, int c1,int c2,Mat& meansInitial, vector<Mat>& covsInitial, Mat& weightsInitial){
	int k = means.rows;
	int index = 0;

	for(int i=0;i<k;i++){
		if( (i!=c1) && (i!=c2) ){
			means.row(i).copyTo(meansInitial.row(index));
			weightsInitial.at<double>(0,index) = weights.at<double>(0,i);
			index++;
			covsInitial.push_back(covs[i]);
		}
	}
	double weightC1 = weights.at<double>(0,c1);
	double weightC2 = weights.at<double>(0,c2);
	weightsInitial.at<double>(0,index) = weightC1 + weightC2;

	Mat newMean = ( weightC1*means.row(c1) + weightC2*means.row(c2) )/(weightC1+weightC2);
	newMean.copyTo(meansInitial.row(index));

	Mat meanC1diff = means.row(c1) - newMean;
	Mat meanC2diff = means.row(c2) - newMean;
	Mat newCov = ( weightC1*(covs[c1]+meanC1diff.t()*meanC1diff) 
	              + weightC2*(covs[c2]+meanC2diff.t()*meanC2diff))/(weightC1+weightC2);
	covsInitial.push_back(newCov);
}

//Distance between two components as per CLUSTER algorithm
double distance(const Mat& mean1, const Mat& mean2, const Mat& cov1, const Mat& cov2, double w1, double w2, int n){
	Mat newMean = ( w1*mean1 + w2*mean2 )/(w1+w2);
	
	Mat meanC1diff = mean1 - newMean;
	Mat meanC2diff = mean2 - newMean;
	Mat newCov = ( w1*(cov1+meanC1diff.t()*meanC1diff) 
	              + w2*(cov2+meanC2diff.t()*meanC2diff))/(w1+w2);

	double d = 0.5 * n * ( w1 * log(determinant(newCov)/determinant(cov1)) + 
	                       w2 * log(determinant(newCov)/determinant(cov2)) );

	return d;
}

//Find a pair c1,c2 with produce least MDL change
void leastMdlChange(const Mat& means, const Mat& weights, const vector<Mat>& covs, int& c1, int& c2, int n){
	int k = means.rows;
	double dmin = numeric_limits<double>::max();

	for(int i=0; i<k-1; i++){
		for(int j=i+1; j<k; j++){
			double d = distance(means.row(i), means.row(j), covs[i], covs[j],
			                    weights.at<double>(0,i), weights.at<double>(0,j),n);
			if( d<dmin ){
				dmin = d;
				c1 = i;
				c2 = j;
			}
		}
	}
	
}


int emMdlBd(const Mat& samples, string imgName, string fgdbgd, Mat& minMeans, Mat& minWeights, vector<Mat>& minCovs)
{
	ofstream mdl_output;
	string mdl_file = "output/mdl/" + imgName + fgdbgd;
	mdl_file.erase(mdl_file.length()-4);
	mdl_output.open(mdl_file.c_str());
		
	Mat meansInitial;
	Mat weightsInitial;
	vector<Mat> covsInitial;

	double minMdl = numeric_limits<double>::max();
	int minK;
	const int maxComponents = 10;
	const int minComponents = 1;
	
	for(int k=maxComponents; k>=minComponents; k--){
		cout << "Components " << k << endl;
			
		EM model(k,EM::COV_MAT_GENERIC);
				
		Mat labels;
		Mat probs;
    	Mat log_likelihoods;
         
    	if( k==maxComponents )
			model.train(samples,log_likelihoods, labels, probs);
		else
			model.trainE(samples, meansInitial, covsInitial, weightsInitial, log_likelihoods, labels, probs);
		
		
		//Get estimated means, weights and covariances 
		int c1,c2;
		Mat means = model.get<Mat>("means");
		Mat weights = model.get<Mat>("weights");
		vector<Mat> covs = model.get<vector<Mat> >("covs");

		//Calculate sum of all likelihoods
		double total_likelihood = 0.0;
		for(int i=0;i<log_likelihoods.rows;i++){
			double t = log_likelihoods.at<double>(i,0);
			total_likelihood += t;			
		}

		//Calculate penalty term according to MDL criteria
		int dimension =3;
		double l = k*(1 + dimension + ((dimension+1)*dimension)/2)-1;
		double penalty = 0.5*l*log(samples.rows*dimension);

		//Add the Mixture-MDL penalty term
		double logweights = 0.0;
		for(int i=0; i<k; i++){
			logweights += log(weights.at<double>(0,i));
		}
		penalty = penalty + 0.5*(dimension + dimension*(dimension+1)/2) * logweights;

		double mdl = -total_likelihood + penalty;
		
		if(mdl < minMdl)
		{	
			minMdl = mdl;
			minK = k;
			means.copyTo(minMeans);
			weights.copyTo(minWeights);
			minCovs.clear();
			for(int i=0; i<covs.size(); i++){
				minCovs.push_back(covs[i]);
			}
		}
		
		if( k > minComponents ){

			meansInitial = Mat(means.rows-1,means.cols,means.type());
			weightsInitial = Mat(1,weights.cols-1,weights.type());
			covsInitial.clear();
			
			if( k==minComponents+1 )
				mergeComponents(means,covs,weights,0,1,meansInitial,covsInitial,weightsInitial);
			else{
					leastBhattacharyaDist(means,covs,c1,c2);
					mergeComponents(means,covs,weights,c1,c2,meansInitial,covsInitial,weightsInitial);
			}
		}

		mdl_output << endl << "********** No. of components=" << k << "***********" << endl;
		mdl_output << "Penalty=" << penalty << endl;
		mdl_output << "Total log likelihood=" << fixed << total_likelihood << endl;
		mdl_output << "MDL=" << fixed << mdl << endl;
	}
	mdl_output << endl << "***********Result***********" << endl;
	mdl_output << "Min MDL =" << minMdl << endl;
	mdl_output << "Components =" << minK; 
	mdl_output.close();

	return minK;
}

int main(int argc, char* argv[]){

	//Read data for foreground
	ifstream data_file("input/data/fgd");
	if(!data_file){
		cout << "Could not open fgd file" << endl;
		return -1;
	}

	vector<Vec3d> vSamples;

	string line;
	while( getline(data_file,line) ){
		stringstream s(line);
		Vec3d v;
		s >> v[0] >> v[1] >> v[2];
		vSamples.push_back(v);
	}

	data_file.close();

	Mat samples((int)vSamples.size(),3,CV_64FC1,&vSamples[0][0]);

	Mat minMeans, minWeights;
	vector<Mat> minCovs;
	cout << "Estimating parameters for foreground GMM..." << endl;
	int kFgd = emMdlBd(samples, argv[1], "fgd", minMeans, minWeights, minCovs);

	//Estimated parameters are stored in "input/params/fgd.xml"
	FileStorage fs("input/params/fgd.xml",FileStorage::WRITE);
	fs << "Components" << kFgd;
	fs << "Means" << minMeans;
	fs << "Weights" << minWeights;
	fs << "Covs" << minCovs;
	vSamples.clear();

	//Get data for background from file "input/data/bgd" 
	data_file.open("input/data/bgd");
	if(!data_file){
		cout << "Could not open bgd file" << endl;
		return -1;
	}

	while( getline(data_file,line) ){
		stringstream s(line);
		Vec3d v;
		s >> v[0] >> v[1] >> v[2];
		vSamples.push_back(v);
	}
	data_file.close();

	samples = Mat((int)vSamples.size(),3,CV_64FC1,&vSamples[0][0]);
	cout << "Estimating parameters for background GMM..." << endl;
	int kBgd = emMdlBd(samples, argv[1], "bgd", minMeans, minWeights, minCovs);

	//Store estimated parameters
	fs = FileStorage("input/params/bgd.xml",FileStorage::WRITE);
	fs << "Components" << kBgd;
	fs << "Means" << minMeans;
	fs << "Weights" << minWeights;
	fs << "Covs" << minCovs;

	ofstream out;
	string acc_file("output/result");
	out.open(acc_file.c_str(), ios::app);	
	out << argv[1] << "\t" << kFgd << "\t" << kBgd << "\t";
}