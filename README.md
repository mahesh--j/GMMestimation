GMMestimation
=============

Implements the algorithm described in "Cluster- Unsupervised algorithm for modelling Gaissian mixtures" (by CA Bouman)
The penalty term was modified as per the description in "On fitting mixture models"(by MAT Figueiredo)
The implementation was used to work with the Grabcut algorithm for image segmentation, so there is a estimation 
of foreground GMM and background GMM.
The create_data_file.cpp takes as input the Image file and the co-ordinates of the rectangle drawn around the objecct to 
be segmented. It separately stores the foreground and background pixels.
The learnGMM.cpp reads the foreground and background data and estimates the GMM parameters as per the Cluster algorithm.
