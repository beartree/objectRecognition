//
//  main.cpp
//  cvhw5
//
//  Created by Xiong Shu on 11/19/13.
//  Copyright (c) 2013 Xiong Shu. All rights reserved.
//



#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>


using namespace cv;
using namespace std;

//get All the filenames under the file folder
void listFile(char * dir, vector<string> &files){
    DIR *pDIR;
    struct dirent *entry;
    if( pDIR=opendir(dir) ){
        while(entry = readdir(pDIR)){
            if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 && strcmp(entry->d_name, ".DS_Store") != 0)
                files.push_back((string)dir +"/"+ entry->d_name);
        }
        closedir(pDIR);
    }
}

//get the sift features of an image
void getSIFT(vector<string> files, vector<Mat> &desc, int minHessian = 50){
    
    for (vector<string>::iterator it = files.begin() ; it != files.end(); ++it){
        Mat img = imread(*it, CV_LOAD_IMAGE_GRAYSCALE);
        SiftFeatureDetector detector( minHessian );
        vector<KeyPoint> keypoints;
        detector.detect(img, keypoints);
        SIFT sift;
        Mat descriptors;
        sift.operator()(img, cv::noArray(), keypoints, descriptors, true);
        desc.push_back(descriptors);
    }
}

//PCA for sift features
Mat pca_sift(Mat desc, Mat &pcadata, int compo_num, Mat &mean){

        Mat covar, mean1, eigenvalues, eigenvectors, featurevector, adjustdata;
        calcCovarMatrix(desc, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        eigen(covar, eigenvalues, eigenvectors);
        cout << eigenvectors.type() << endl;
        featurevector = Mat(compo_num, eigenvectors.cols, desc.type());
        for (int i = 0; i < compo_num; i++) {
            eigenvectors.row(i).copyTo(featurevector.row(i));
        }
        mean1 = Mat(desc.rows, desc.cols, desc.type());
        adjustdata = Mat(desc.rows, desc.cols, desc.type());
        for (int i = 0; i < desc.rows; i++) {
            mean.row(0).copyTo(mean1.row(i));
        }
        subtract(desc, mean1, adjustdata);
        transpose(adjustdata, adjustdata);
        pcadata = featurevector * adjustdata;
        transpose(pcadata, pcadata);
        return featurevector;
}

//merge all the extracted sift features into one matrix
void merge_pcasift(int &i, vector<Mat> pcasift, Mat &pcasift_all, int minHessian){
    for (vector<Mat>::iterator it = pcasift.begin() ; it != pcasift.end(); ++it){
        Mat descriptor = *it;
        int upbound = 0;
        if (minHessian > descriptor.rows) {
            upbound = descriptor.rows;
        }else{
            upbound = minHessian;
        }
        for (int j = 0; j < upbound; j++) {
            descriptor.row(j).copyTo(pcasift_all.row(i));
            i++;
        }
        if (upbound != minHessian) {
            for (int j = 0; j < minHessian - descriptor.rows; j++) {
                if (j < descriptor.rows) {
                    descriptor.row(j).copyTo(pcasift_all.row(i));
                }else{
                    descriptor.row(0).copyTo(pcasift_all.row(i));
                }
                
                i++;
            }
        }
    }
    
}


//compuate the histogram feature vector for each image
void computeHist(Mat &hist, Mat labels, int feature_num, int img_num){
    for (int i = 0; i < feature_num; i++) {
        for (int j = 0; j < img_num; j++) {
           
                hist.at<int>(j,labels.at<int>(i+j*feature_num,0))++;
        }
    }
    
}

void computeHist(Mat &hist, int *labels, int feature_num, int img_num){
    for (int i = 0; i < feature_num; i++) {
        for (int j = 0; j < img_num; j++) {
            hist.at<int>(j,labels[i+j*feature_num])++;
        }
    }
    
}

//compuate the feature vector for test image
void computeFV(Mat pcasift, Mat centroid, Mat &FeatureVector, int minHessian){
    int *labels = (int*)calloc(pcasift.rows, sizeof(int));
    
    for (int h = 0; h < pcasift.rows; h++) {
        double min_distance = DBL_MAX;
        for (int i = 0; i < centroid.rows; i++) {
            double distance = 0;
            for (int j = 0; j < centroid.cols; j++) {
                distance += pow(pcasift.at<float>(h,j) - centroid.at<float>(i,j), 2);
            }
            if (distance < min_distance) {
                labels[h] = i;
                min_distance = distance;
            }
        }
    }
    
    computeHist(FeatureVector, labels, pcasift.rows, 1);
}

struct Distance
{
    int distance;
    int index;
};

bool compareDistance(const Distance &a, const Distance &b){
    return a.distance < b.distance;
}

//modified distance weighted knn classifier
int knn(int k, Mat featurevector, Mat hist){
    int min_distance = 1000000;
    int min_indx = -1;
    
    vector<Distance> distancelist;
    for (int i = 0; i < hist.rows; i++) {
        int distance = 0;
        for (int j = 0; j < hist.cols; j++) {
            distance += (featurevector.at<int>(0,j) - hist.at<int>(i,j))*(featurevector.at<int>(0,j) - hist.at<int>(i,j));
            
        }
        Distance obj;
        obj.distance = distance;
        obj.index = i;
        distancelist.push_back(obj);
    }
    sort(distancelist.begin(), distancelist.end(), compareDistance);
    int i = 0;
    double category[5] = {0.0};
    for (vector<Distance>::iterator it = distancelist.begin(); i < k; ++it,i++) {
        Distance dis = *it;
        int idx = dis.index / 20;
        category[idx] += double( 1.0 / double(dis.distance));
    }
    double max = -1.0;
    int cat_indx = -1;
    for (i = 0; i < 5; i++) {
        if (category[i] > max) {
            cat_indx = i;
            max = category[i];
        }
    }
    for (int i = 0; i < featurevector.rows; i++) {
        for (int j = 0; j < featurevector.cols; j++) {
            cout << featurevector.at<int>(i,j) << " ";
        }
        cout << endl;
    }
    
    cout << cat_indx << endl;
    return cat_indx;
}

/** @function main */
int main( int argc, char** argv ){

    int compo_num = 15;
    int minHessian = 200;
    int cluster_num = 50;
    int k_nb = 8;
    
    //get siftpca features for car
    char *dir_car = "HW5_data/training/car";
    vector<string> files_car;
    vector<Mat> siftFeatures_car;
    listFile(dir_car, files_car);
    getSIFT(files_car, siftFeatures_car, minHessian);
    
    //get siftpca features for face
    char *dir_face = "HW5_data/training/face";
    vector<string> files_face;
    vector<Mat> siftFeatures_face;
    listFile(dir_face, files_face);
    getSIFT(files_face, siftFeatures_face,minHessian);
    
    //get siftpca features for laptop
    char *dir_laptop = "HW5_data/training/laptop";
    vector<string> files_laptop;
    vector<Mat> siftFeatures_laptop;
    listFile(dir_laptop, files_laptop);
    getSIFT(files_laptop, siftFeatures_laptop,minHessian);
    
    //get siftpca features for motorbike
    char *dir_motorbike = "HW5_data/training/motorbike";
    vector<string> files_motorbike;
    vector<Mat> siftFeatures_motorbike;
    listFile(dir_motorbike, files_motorbike);
    getSIFT(files_motorbike, siftFeatures_motorbike, minHessian);

    
    //get siftpca features for pigeon
    char *dir_pigeon = "HW5_data/training/pigeon";
    vector<string> files_pigeon;
    vector<Mat> siftFeatures_pigeon;
    listFile(dir_pigeon, files_pigeon);
    getSIFT(files_pigeon, siftFeatures_pigeon, minHessian);
    
    
    Mat sift_all = Mat(siftFeatures_pigeon.size()* minHessian * 5, 128, siftFeatures_pigeon.front().type());
    
    //Mat sift_all;
    
    int index = 0;
    
    merge_pcasift(index, siftFeatures_car, sift_all, minHessian);
    merge_pcasift(index, siftFeatures_face, sift_all, minHessian);
    merge_pcasift(index, siftFeatures_laptop, sift_all, minHessian);
    merge_pcasift(index, siftFeatures_motorbike, sift_all, minHessian);
    merge_pcasift(index, siftFeatures_pigeon, sift_all, minHessian);
    
    Mat sift_all1 = Mat :: zeros(index, 128, sift_all.type());
    for (int i = 0; i < index; i++) {
        for (int j = 0; j < 128; j++) {
            sift_all1.at<float>(i,j) = sift_all.at<float>(i,j);
        }
    }
    
    Mat pcasift = Mat(sift_all1.rows, compo_num, sift_all1.type());

    Mat mean;
    Mat Wpca = pca_sift(sift_all1, pcasift, compo_num, mean);

    Mat labels, centroids;
    TermCriteria epsilon = TermCriteria();
    kmeans(pcasift, cluster_num, labels, epsilon, 20, KMEANS_PP_CENTERS, centroids);

    
    pcasift.release();
    Mat hist = Mat :: zeros(siftFeatures_car.size() * 5, cluster_num, CV_32S);
    computeHist(hist, labels, minHessian, siftFeatures_car.size() * 5);
    
    //get siftpca features for test car
    char *dir_test_car = "HW5_data/testing/car";
    vector<string> files_test_car;
    vector<Mat> siftFeatures_test_car;
    listFile(dir_test_car, files_test_car);
    getSIFT(files_test_car, siftFeatures_test_car, minHessian);
    
    transpose(Wpca, Wpca);
    
    for (vector<Mat>::iterator it = siftFeatures_test_car.begin() ; it != siftFeatures_test_car.end(); ++it) {
        Mat sift_features = *it;
        Mat mean1 = Mat(sift_features.rows, sift_features.cols, sift_features.type());
        for (int i = 0; i < sift_features.rows; i++) {
            mean.row(0).copyTo(mean1.row(i));
        }
        subtract(sift_features, mean1, sift_features);
        Mat pcasift_features = sift_features * Wpca;
        Mat featurevector = Mat :: zeros(1, cluster_num, CV_32S);
        computeFV(pcasift_features, centroids, featurevector, minHessian);
        knn(k_nb, featurevector, hist);
        
    }
    
    for (int i = 0; i < hist.rows; i++) {
        for (int j = 0; j < hist.cols; j++) {
            cout << hist.at<int>(i,j) << " ";
        }
        cout << endl;
    }
    return 0;
}

