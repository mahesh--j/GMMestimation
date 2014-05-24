/* Generate samples from uniform distibution with predefined parameters
   Estimates parameters of the samples using Expectation maximization along with 
   Minimum description length principle and Bhatacharyya distance measure */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <list>

using namespace std;
using namespace cv;

double BhattacharyaDist(const Mat& mean1,const Mat& mean2,const Mat& cov1,const Mat& cov2)
{
	Mat avgCov = (cov1 + cov2)/2;
	Mat mDiff = mean1-mean2;
	Mat temp = mDiff*avgCov.inv()*mDiff.t();
	double term1 = temp.at<double>(0,0)/8;
	double term2 = 0.5*log(determinant(avgCov)/(sqrt(determinant(cov1)*determinant(cov2))));
	return term1+term2;
	
}

void leastBhattacharyaDist(Mat means, vector<Mat> covs, int& c1,int& c2)
{
	ofstream distFile;
	string fileName("./output/dist");
	int k=means.rows;
	stringstream s;
	s << k;
	fileName = fileName + s.str();
	distFile.open(fileName.c_str());
	
	double minDist = numeric_limits<double>::max();
	for(int i=0;i<k-1;i++)
	{
		for(int j=i+1;j<k;j++)
		{
			double dist = BhattacharyaDist(means.row(i),means.row(j),covs[i],covs[j]);
			distFile << i <<"\t" << j << "\t" << dist << endl; 
			if( dist<minDist )
			{
				minDist = dist;
				c1 = i;
				c2 = j;
			}
		}
	}
	distFile << "Least distance between" << c1 << "and" << c2 << endl;
}	

void mergeComponents(const Mat& means, const vector<Mat>& covs, const Mat& weights, int c1,int c2,Mat& meansCombine, vector<Mat>& covsCombine, Mat& weightsCombine){
	int k = means.rows;
	int index = 0;

	for(int i=0;i<k;i++){
		if( (i!=c1) && (i!=c2) ){
			means.row(i).copyTo(meansCombine.row(index));
			weightsCombine.at<double>(0,index) = weights.at<double>(0,i);
			index++;
			covsCombine.push_back(covs[i]);
		}
	}
	double weightC1 = weights.at<double>(0,c1);
	double weightC2 = weights.at<double>(0,c2);
	weightsCombine.at<double>(0,index) = weightC1 + weightC2;

	Mat newMean = ( weightC1*means.row(c1) + weightC2*means.row(c2) )/(weightC1+weightC2);
	newMean.copyTo(meansCombine.row(index));

	Mat meanC1diff = means.row(c1) - newMean;
	Mat meanC2diff = means.row(c2) - newMean;
	Mat newCov = ( weightC1*(covs[c1]+meanC1diff.t()*meanC1diff) + weightC2*(covs[c2]+meanC2diff.t()*meanC2diff))/(weightC1+weightC2);
	covsCombine.push_back(newCov);
}

int emMdl(const Mat& samples,Mat& optimumMeans, Mat& optimumWeights, vector<Mat>& optimumCovs)
{
	string mdl_file = string("./output/emMdl/mdl_output");
	string log_file = string("./output/emMdl/log_output");
	
	ofstream mdl_output, log_output;
	mdl_output.open(mdl_file.c_str());
	log_output.open(log_file.c_str());

	mdl_output << "MDL criteria::" << endl;
	
	Mat meansCombine;
	Mat weightsCombine;
	vector<Mat> covsCombine;

	int minComponents = 1;
	int maxComponents = 15;
	int optimumComponents;
	double minMdl = numeric_limits<double>::max();
	
	for(int k=maxComponents; k>=minComponents; k--){

		EM model(k,EM::COV_MAT_GENERIC);
				
		Mat labels;
		Mat probs;
    	Mat log_likelihoods;
		
    	if( k==maxComponents )
			model.train(samples,log_likelihoods, labels, probs);
		else
			model.trainE(samples, meansCombine, covsCombine, weightsCombine, log_likelihoods, labels, probs); //provide parameters as per previous iteration results
	
		
		Mat means = model.get<Mat>("means");
		Mat weights = model.get<Mat>("weights");
		vector<Mat> covs = model.get<vector<Mat> >("covs");

		double total_likelihood = 0.0;
		log_output << endl <<"********** No. of components=" << k << "***********" << endl;
		
		for(int i=0;i<log_likelihoods.rows;i++){
			double t = log_likelihoods.at<double>(i,0);
			log_output << t << endl;
			total_likelihood += t;	
		}

		string p("parameters");
		stringstream os;
		os << k;
		string path = string("./output/emMdl/") + string("/parameters/") + p + os.str();
		ofstream p_file( path.c_str() );
		p_file << "Gaussian parameters for k= " << k << endl << endl;

		for(int i=k-1; i>=0; i--){
			p_file << "******Component " << i <<"*********" << endl;
			p_file << "Mean:" << means.row(i) << endl;
			p_file << "Covariance matrix:" << endl << covs[i] << endl;
			p_file << "Weight: " << weights.col(i) << endl << endl;
		}
				
    	int dimension = 3;
		double l = k*(1 + dimension + ((dimension+1)*dimension)/2)-1;
		double penalty = 0.5*l*log(samples.rows*dimension);


		double logweights = 0.0;
		for(int i=0; i<k; i++){
			logweights += log(weights.at<double>(0,i));
		}
		penalty = penalty + 0.5*(dimension + dimension*(dimension+1)/2) * logweights;

		
		double mdl = -total_likelihood + penalty;
		mdl_output << endl << "********** No. of components=" << k << "***********" << endl;
		mdl_output << "Penalty=" << penalty << endl;
		mdl_output << "Total log likelihood(EM result)=" << fixed << total_likelihood << endl;
		mdl_output << "MDL result=" << fixed << mdl << endl;

		if(mdl < minMdl)
		{	
			minMdl = mdl;
			optimumComponents = k;
			means.copyTo(optimumMeans);
			weights.copyTo(optimumWeights);
			optimumCovs.clear();
			for(int i=0; i<covs.size(); i++)
				optimumCovs.push_back(covs[i]);
		}
		
		int c1,c2;
		if( k>1 ){
			leastBhattacharyaDist(means,covs,c1,c2);
			mdl_output << "Merging components: " << c1 <<" and " << c2 <<endl;

			meansCombine = Mat(means.rows-1,means.cols,means.type());
			weightsCombine = Mat(1,weights.cols-1,weights.type());
			covsCombine.clear();
				
			mergeComponents(means,covs,weights,c1,c2,meansCombine,covsCombine,weightsCombine);
		}
	}
	mdl_output << endl << "***********Result***********" << endl;
	mdl_output << "Minimum MDL value=" << minMdl << endl;
	mdl_output << "Optimal number of components=" << optimumComponents;
	mdl_output.close();

	return optimumComponents;
}

int main(int argc, char*argv[]){

	RNG rngMean, rngCov, rngWeight ;
	int k = 7;
	int dimension = 3;
	int n = 1000;
	

	Mat means(k, dimension, CV_64FC1); //each row is a mean for component i=0 to k-1
	list<Mat> covs, _covs;
	Mat cur_cov(dimension, dimension, CV_64FC1);
	Mat weights(1, k, CV_64FC1);
	double sumWeights = 0.0;
	
	for(int i=0; i<k; i++){
		rngMean.fill(means.row(i), RNG::UNIFORM, 0, 256);
		rngCov.fill(cur_cov, RNG::UNIFORM, 50, -50);	
		covs.push_back(cur_cov);
		_covs.push_back(cur_cov);
		double randWeight = rngWeight.uniform( (double)0, (double)1/k );
		if( randWeight < (double)1/(2*k) )
			weights.at<double>(0,i) = (double)1/k - randWeight;
		else
			weights.at<double>(0,i) = randWeight;
		if( i!=k-1 )
			sumWeights += weights.at<double>(0,i);
	}
	weights.at<double>(0,k-1) = 1-sumWeights;

	cout << "Actual mixture parameters::" << endl;
	cout << "Number of components::" << k << endl;
	cout << "Means::" << endl << means << endl;
	cout << "Weights::" << endl << weights << endl;
	
	vector<Vec3d> vecSamples;
	Mat samplesk;
	int sizek;
	for(int i=0; i<k; i++){
		sizek = (int)n*weights.at<double>(0,i);
		samplesk = Mat(sizek, 1, CV_64FC3);
		RNG rng;
		rng.fill(samplesk, RNG::NORMAL, means.row(i), _covs.front());
		_covs.pop_front();

		for(int j=0; j<sizek; j++)
			vecSamples.push_back( samplesk.at<Vec3d>(j,0) );		
	}

	Mat samples((int)vecSamples.size(), 3, CV_64FC1);

	for(int i=0; i<samples.rows; ++i){
		for(int j=0; j<samples.cols; ++j){
			samples.at<double>(i,j) = vecSamples[i][j];
		}
	}
	
	Mat optimumWeights, optimumMeans;
	vector<Mat> optimumCovs;
		
	int kOptimum = emMdl(samples,optimumMeans,optimumWeights,optimumCovs);

	cout << endl << "Estimated parameters::" << endl;
	cout << "Number of components::" << kOptimum << endl;
	cout << "Means::" << optimumMeans << endl;
	cout << "Weights::" << optimumWeights << endl;
		
}