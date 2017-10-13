////////////////////////////////////////////////////////////////////
//
// 2017 Oct 12th
// Yoshi Ri @ Univ Tokyo
// Tested in Jetson TX1 with Ubuntu 16 and Opencv3.1.0 with cuda8.0
// Reference    //http://www.coldvision.io/2016/06/27/object-detection-surf-knn-flann-opencv-3-x-cuda/
//
// compile example :$ g++ -ggdb TempTrackGPU.cpp -o temptrackinggpu `pkg-config --cflags --libs opencv`
//
// Usage : press 'q' to quit, press 's' to save current image
////////////////////////////////////////////////////////////////////


// for image processing
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/core.hpp"
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int64 prev,current;

class TempTracker_GPU
{
private:
    
    cv::Mat _temp,_show;
	// Copy the image into GPU memory
	cuda::GpuMat _img1,_img2;
    cuda::GpuMat _gkp1, _gkp2;
	vector< KeyPoint > _kp1, _kp2;
    cuda::GpuMat _des1, _des2;
    cuda::SURF_CUDA _surf;
    Ptr< cuda::DescriptorMatcher > _matcher;
	vector< vector< DMatch> > _matches;
	cv::Mat _H;
	std::vector<Point2f> _obj_corners;
	std::vector<Point2f> _scene_corners;
    
public:
    //TempTracker_GPU();
    ~TempTracker_GPU();
    void LoadTemp(cv::Mat);
    void track(cv::Mat);
    void get_rect();
    void saveimg();
};

// constructor
//TempTracker_GPU::TempTracker_GPU(){
//}

// Initialize template
void TempTracker_GPU::LoadTemp(cv::Mat templat)
{
    _temp = templat;
    _img1.upload(_temp); //upload template    
    _surf( _img1, cuda::GpuMat(), _gkp1, _des1 ); //extract template surf
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    _matcher = matcher; //define matcher
    _H = cv::Mat::eye(3,3,CV_32FC1); // init Homography

    // for get rect
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0.0, 0.0);
	obj_corners[1] = cvPoint(float(_temp.cols), 0.0);
	obj_corners[2] = cvPoint(float(_temp.cols), float(_temp.rows));
	obj_corners[3] = cvPoint(0.0, float(_temp.rows));
	_obj_corners = obj_corners;
	_scene_corners = obj_corners;	
	//_obj_corners.push(cvPoint(0, 0));
	//_obj_corners.push(cvPoint(_temp.cols, 0));
	//_obj_corners.push(cvPoint(_temp.cols, _temp.rows));
	//_obj_corners.push(cvPoint(0, _temp.rows));
}

// Tracking and show
void TempTracker_GPU::track(cv::Mat now){
    _show = now.clone();
    _img2.upload(now); //upload
    _surf( _img2, cuda::GpuMat(), _gkp2, _des2 ); //extract template surf
    _matcher->knnMatch(_des1, _des2, _matches, 2);//matching
    
	_surf.downloadKeypoints(_gkp1, _kp1);//download kp
	_surf.downloadKeypoints(_gkp2, _kp2);
    
    // pick good matches
	std::vector< DMatch > good_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    int count = 0;
	for (int k = 0; k < std::min(_kp1.size()-1, _matches.size()); k++)
	{
		if ( (_matches[k][0].distance < 0.6*(_matches[k][1].distance)) &&
				((int)_matches[k].size() <= 2 && (int)_matches[k].size()>0) )
		{
			// take the first result only if its distance is smaller than 0.6*second_best_dist
			// that means this descriptor is ignored if the second distance is bigger or of similar
			good_matches.push_back(_matches[k][0]);
            obj.push_back( _kp1[ _matches[k][0].queryIdx ].pt );
            scene.push_back( _kp2[ _matches[k][0].trainIdx ].pt );
            count++;
		}
	}
	printf("Matched %d points!\n",count);	
	if (count < 12){
	    return; // not enough matching points
	}
	
	cv::Mat mask;		
	// Extract Homography and Tracking
    _H = findHomography( obj, scene, mask, RANSAC, 3.0);
    
    // Count inliers
    int inlier = cv::countNonZero(mask);
	printf("Inlier is %d points!\n",inlier);
    if(inlier < count * 0.4){
        return; // If inlier is under 40 percent of good matches...
    }
    // Show rectangle
    TempTracker_GPU::get_rect();
    cv::imshow("tracked",_show);
}


void TempTracker_GPU::get_rect()
{
    printf("Tracking...\n");
	perspectiveTransform(_obj_corners, _scene_corners, _H);
	
	// Draw lines between the corners (the mapped object in the scene - image_2 )
	line(_show, _scene_corners[0],
			_scene_corners[1] ,
			Scalar(255, 0, 0), 4);
	line(_show, _scene_corners[1],
			_scene_corners[2] ,
			Scalar(255, 0, 0), 4);
	line(_show, _scene_corners[2] ,
			_scene_corners[3] ,
			Scalar(255, 0, 0), 4);
	line(_show, _scene_corners[3],
			_scene_corners[0] ,
			Scalar(255, 0, 0), 4);

}

// save image
void TempTracker_GPU::saveimg()
{
    imwrite("results.png",_show);
}

// Destracter
TempTracker_GPU::~TempTracker_GPU()
{
	//-- Release Memory
	_surf.releaseMemory();
	_matcher.release();
	_img1.release();
	_img2.release();
	_gkp1.release();
	_gkp2.release();
	_des1.release();
	_des2.release();
}





int main(){

    printf("...Cuda initialization... This may take around 30 sec ...");

    cv::Mat temp=cv::imread("object.png",0);
    cv::imshow("template",temp);

    //dummy function to call cuda 
    cv::cuda::GpuMat test;
    test.create(1, 1, CV_8U);


    cv::VideoCapture cap(0);

    if(!cap.isOpened()){  // check if we succeeded
        printf("can not open video !!");
        return -1;
    }

    // get first frame
    cv::Mat frame,gray;
    cap >> frame; // get a new frame from camera
    cv::waitKey(1);

    //Init tracker
    TempTracker_GPU tracker;
    tracker.LoadTemp(temp);
    // While cap is opened
    while(cap.isOpened())
    {
        prev = cv::getTickCount();
        cap >> frame; // get a new frame from camera
        imshow("now",frame);        
        cvtColor(frame, gray, CV_BGR2GRAY);
        tracker.track(gray);
                
        char ch;
        ch = waitKey(1);
        if(ch == 'q'){
            break;
        }else if(ch == 's'){
            tracker.saveimg();
        }
        // measuring time
        current = cv::getTickCount();
        std::cout << "Computation takes..." << (current-prev)/cv::getTickFrequency() << " [s]" <<std::endl;
        prev = current;
    }
    
    printf("Finish !!");
    
}

