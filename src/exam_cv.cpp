#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stack>
#include <string>
#include "util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define PI 3.1415926

using namespace std;
using namespace cv;

const Scalar COLOR_BLUE = Scalar(255, 0, 0);
const Scalar COLOR_RED = Scalar(0, 0, 255);
const Scalar COLOR_GREEN = Scalar(170, 170, 0);


const Vec3b RGB_WHITE_LOWER = Vec3b(100, 100, 180);
const Vec3b RGB_WHITE_UPPER = Vec3b(255, 255, 255);

const Vec3b HSV_YELLOW_LOWER = Vec3b(10, 70, 130);
const Vec3b HSV_YELLOW_UPPER = Vec3b(50, 255, 255);

const Vec3b HSV_WHITE_LOWER = Vec3b(80, 0, 180);
const Vec3b HSV_WHITE_UPPER = Vec3b(180, 60, 255);

const Vec3b HSV_RED_LOWER = Vec3b(0, 150, 100);
const Vec3b HSV_RED_UPPER = Vec3b(10, 255, 255);
const Vec3b HSV_RED_LOWER1 = Vec3b(160, 100, 100);
const Vec3b HSV_RED_UPPER1 = Vec3b(180, 255, 255);

const Vec3b HSV_GREEN_LOWER = Vec3b(50, 80, 50);
const Vec3b HSV_GREEN_UPPER = Vec3b(100, 255, 255);

const Vec3b HSV_BLACK_LOWER = Vec3b(0, 0, 0);
const Vec3b HSV_BLACK_UPPER = Vec3b(180, 255, 50);

const Vec3b HSV_OBJECT_LOWER = Vec3b(30,30,30);
const Vec3b HSV_OBJECT_UPPER = Vec3b(220,220,220);

const Vec3b YUV_LOWER = Vec3b(0, 110, 120);
const Vec3b YUV_UPPER = Vec3b(40, 130, 140);

bool get_intersectpoint(const Point& AP1, const Point& AP2,	const Point& BP1, const Point& BP2, Point* IP);
float get_slope(const Point& p1, const Point& p2);
bool hough_left(Mat& img, Mat& srcRGB, Point* p1, Point* p2);
bool hough_right(Mat& img, Mat& srcRGB, Point* p1, Point* p2);
bool hough_curve(Mat& img, Mat& srcRGB, Point* p1, Point* p2);
int curve_detector(Mat& leftImg, Mat& rightImg, int number);
bool hough_horizon(Mat& img, Mat& srcRGB, Point* p1, Point* p2);
float data_transform(float x, float in_min, float in_max, float out_min, float out_max);
void get_center_point(Mat& binaryImg, Point * p);
int Labelling(Mat &img);

/*******************신호등 전역변수*********************/
Point v[2];
bool check[2] = { false, };
int cnt = 0;
/////////////////////////////////////////////////////

struct Object {
    int minx, miny, maxx, maxy;
    int width, height;
    int number;
public: Object(int minx, int miny, int maxx, int maxy, int number) : minx(minx), miny(miny), maxx(maxx), maxy(maxy), number(number) {

}
};

extern "C" {

  int outbreak(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh){
    int redCount = 0;

    Mat dstRGB(nh, nw, CV_8UC3, outBuf);
    Mat srcRGB(ih, iw, CV_8UC3, srcBuf);
    Mat resRGB(ih, iw, CV_8UC3);

    Mat hsvImg, binaryImg, binaryImg1;

    cvtColor(srcRGB, hsvImg, CV_BGR2HSV);

    inRange(hsvImg, HSV_RED_LOWER, HSV_RED_UPPER, binaryImg);
    inRange(hsvImg, HSV_RED_LOWER1, HSV_RED_UPPER1, binaryImg1);

    binaryImg = binaryImg | binaryImg1;
    cvtColor(binaryImg, resRGB,CV_GRAY2BGR);
   resize(resRGB, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);
  
    for(int i = 0; i < binaryImg.cols; i++){
      for(int j = 0; j < binaryImg.rows; j++){
        if(binaryImg.at<uchar>(j, i) == 255) redCount++;
      }
    }
    return redCount;

  }
  int line_detector(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh, float slope[], int modeNum){

    int angle = 1500;
    Point p1, p2, p3, p4, p5;

    bool left_error = true;
    bool right_error = true;

  Mat srcRGB(ih, iw, CV_8UC3, srcBuf); //input
  Mat dstRGB(nh, nw, CV_8UC3, outBuf); //output
  Mat resRGB(ih, iw, CV_8UC3); //reuslt

  Mat oriImg;
  Mat leftROI, rightROI, roiImg;
  Mat hsvImg1, hsvImg2;
  Mat binaryImg1, binaryImg2, binaryImg3, binaryImg4;
  Mat cannyImg1, cannyImg2;
  imwrite("src.png",srcRGB);
  leftROI = srcRGB(Rect(0, srcRGB.rows/3 * 2, srcRGB.cols/2, srcRGB.rows/3));
  rightROI = srcRGB(Rect(srcRGB.cols/2, srcRGB.rows/3 * 2, srcRGB.cols/2, srcRGB.rows/3));
  
  cvtColor(leftROI, hsvImg1, CV_BGR2HSV);
  cvtColor(rightROI, hsvImg2, CV_BGR2HSV);

  inRange(hsvImg1, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER, binaryImg1);
  inRange(hsvImg2, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER, binaryImg2);
   imwrite("test.png",srcRGB);
  
  if(modeNum == 1 || modeNum == 2){ // 2 : Curve mission
    hconcat(binaryImg1, binaryImg2, resRGB);
    cvtColor(resRGB, resRGB, CV_GRAY2BGR);
  }
  else if(modeNum == 3 || modeNum == 4 || modeNum == 5){
    inRange(leftROI, RGB_WHITE_LOWER, RGB_WHITE_UPPER, binaryImg3);
    inRange(rightROI, RGB_WHITE_LOWER, RGB_WHITE_UPPER, binaryImg4);

    binaryImg1 = binaryImg1 | binaryImg3;
    binaryImg2 = binaryImg2 | binaryImg4;

    hconcat(binaryImg1, binaryImg2, resRGB);

    cvtColor(resRGB, resRGB, CV_GRAY2BGR);
    // resize(resRGB, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);
    // imwrite("t1.png",dstRGB);
  }
  Canny(binaryImg1, cannyImg1, 150, 250);
  Canny(binaryImg2, cannyImg2, 150, 250);

    
  left_error = hough_left(cannyImg1, leftROI, &p1, &p2);
  right_error = hough_right(cannyImg2, rightROI, &p3, &p4);


  if(left_error || right_error){

    angle = curve_detector(cannyImg1, cannyImg2, modeNum);
  }
  else{

    get_intersectpoint(p1, p2, Point(p3.x + 160, p3.y), Point(p4.x + 160, p4.y), &p5);


    float steer;
    float x_Difference = 160.0 - p5.x;

    if(x_Difference > 0.0){
      steer = 1500.0 + 0.1 * x_Difference;
    }
    else if(x_Difference < 0.0){
      steer = 1500.0 + 0.1 * x_Difference;
    }
    else{
      steer = 1500.0;
    }
    angle = steer;
  }

  // resize(resRGB, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);

  if(angle > 2000){
    angle = 2000;
  }
  else if(angle < 1000){
    angle = 1000;
  }

  return angle;

}

int stop_line_detector(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh){

  Mat srcRGB(ih, iw, CV_8UC3, srcBuf); //input
  Mat dstRGB(nh, nw, CV_8UC3, outBuf); //ouput
  Mat resRGB(ih, iw, CV_8UC3);         //result

  Mat roiImg, hsvImg, binaryImg;

  int cnt = 0;

  roiImg = srcRGB(Rect(0, srcRGB.rows/3 * 2, srcRGB.cols, srcRGB.rows/3));

  cvtColor(roiImg, hsvImg, CV_BGR2HSV);
  inRange(hsvImg, HSV_WHITE_LOWER, HSV_WHITE_UPPER, binaryImg);

  for(int i = 0; i < binaryImg.cols; i++){
    for(int j = 0; j < binaryImg.rows; j++){
      if(binaryImg.at<uchar>(j, i) == 255) cnt ++;
    }
  }

  return cnt;
}

bool is_yellow_horizental(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh, int modeNum, float ab[]){


  Mat srcRGB(ih, iw, CV_8UC3, srcBuf); //input
  Mat dstRGB(nh, nw, CV_8UC3, outBuf); //ouput
  Mat resRGB(ih, iw, CV_8UC3);         //result

  Mat roiImg, hsvImg, binaryImg, cannyImg;
  Point p1, p2;

  bool error = false;
  float slope;

  int cnt = 0;


  slope = get_slope(p1, p2);

  switch (modeNum) {
    case 1 :

    roiImg = srcRGB(Rect(srcRGB.cols/3 * 1, srcRGB.rows/3 * 2 , srcRGB.cols/ 3, srcRGB.rows/3));


    cvtColor(roiImg, hsvImg, CV_BGR2HSV);
    inRange(hsvImg, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER, binaryImg);

    Canny(binaryImg, cannyImg, 150, 250);

    error = hough_horizon(cannyImg, roiImg, &p1, &p2);


    cvtColor(binaryImg, binaryImg, CV_GRAY2BGR);
    // resize(binaryImg, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);

    slope = get_slope(p1, p2);

      if(!error && slope != 0.0 && 0.05 > slope && slope > -0.05){
        return true;
      }
      else
        return false;
        break;
    case 2:

    roiImg = srcRGB(Rect(srcRGB.cols/3, srcRGB.rows/5 * 3, srcRGB.cols/ 3, srcRGB.rows /5  * 2));


    cvtColor(roiImg, hsvImg, CV_BGR2HSV);
    inRange(hsvImg, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER, binaryImg);

    Canny(binaryImg, cannyImg, 150, 250);

    error = hough_horizon(cannyImg, roiImg, &p1, &p2);


    cvtColor(binaryImg, binaryImg, CV_GRAY2BGR);
    //resize(binaryImg, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);

    slope = get_slope(p1, p2);

    ab[0] = slope;

      if(!error){
        return true;
      }
      else
        return false;
      break;
  }


}
int enter_the_rotary(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh){


  Mat srcRGB(ih, iw, CV_8UC3, srcBuf); //input
  Mat dstRGB(nh, nw, CV_8UC3, outBuf); //ouput
  Mat resRGB(ih, iw, CV_8UC3);         //result

  Mat roiImg;
  Mat yuvImg;
  Mat binaryImg;

  int cnt = 0;

  roiImg = srcRGB(Rect(srcRGB.cols/6, srcRGB.rows/3 * 2, srcRGB.cols/6 * 5, srcRGB.rows/3));
  cvtColor(roiImg, yuvImg, CV_BGR2YUV);
  inRange(yuvImg, YUV_LOWER, YUV_UPPER, binaryImg);

  for(int i = 0; i < binaryImg.cols; i++){
    for(int j = 0; j < binaryImg.rows; j++){
      if(binaryImg.at<uchar>(j, i) == 255) cnt ++;
    }
  }
  return cnt;

}


int passing_lane_check(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh, float temp[], float temp2[]){

  Mat srcRGB(ih, iw, CV_8UC3, srcBuf); //input
  Mat dstRGB(nh, nw, CV_8UC3, outBuf); //output
  Mat resRGB(ih, iw, CV_8UC3); //result

  Mat oriImg;
  Mat roiImg;
  Mat yuvImg;
  Mat binaryImg;

  Point p1, p2, p3, p4;
  Point i_p1, i_p2, i_p3, i_p4;

  bool left_error = true;
  bool right_error = true;

  Mat leftROI, rightROI;
  Mat hsvImg1, hsvImg2;
  Mat binaryImg1, binaryImg2, binaryImg3, binaryImg4;
  Mat cannyImg1, cannyImg2;
  cvtColor(srcRGB, hsvImg1, COLOR_BGR2HSV);

  inRange(hsvImg1, HSV_OBJECT_LOWER, HSV_OBJECT_UPPER, binaryImg1);
  //inRange(hsvImg1, HSV_BLACK_LOWER, HSV_BLACK_UPPER, binaryImg2);

  // cannyImg2 = hsvImg1 - cannyImg1;
  // leftROI = srcRGB(Rect(0, srcRGB.rows/3, srcRGB.cols/2, srcRGB.rows/3 * 2));
  // rightROI = srcRGB(Rect(srcRGB.cols/2, srcRGB.rows/ 3, srcRGB.cols/2, srcRGB.rows/3 * 2));

  // cvtColor(leftROI, hsvImg1, CV_BGR2HSV);
  // cvtColor(rightROI, hsvImg2, CV_BGR2HSV);


  // inRange(leftROI, RGB_WHITE_LOWER, RGB_WHITE_UPPER, binaryImg1);
  // inRange(rightROI, RGB_WHITE_LOWER, RGB_WHITE_UPPER, binaryImg2);

  // Canny(binaryImg1, cannyImg1, 150, 250);
  // Canny(binaryImg2, cannyImg2, 150, 250);

  // left_error = hough_left(cannyImg1, leftROI, &p1, &p2);
  // right_error = hough_right(cannyImg2, rightROI, &p3, &p4);




  // temp[0]  = p1.x;
  // temp[1]  = p1.y;
  // temp[2]  = p2.x;
  // temp[3]  = p2.y;
  // temp[4]  = p3.x;
  // temp[5]  = p3.y;
  // temp[6]  = p4.x;
  // temp[7]  = p4.y;

  // int cnt = 0;
  // bool flag;

  // roiImg = srcRGB(Rect(0, srcRGB.rows / 3, srcRGB.cols, srcRGB.rows / 3 * 2));

  // Mat canny_Img;

  // Canny(roiImg, canny_Img, 200, 300);
  // // cvtColor(roiImg, yuvImg, CV_BGR2YUV);

  // // inRange(yuvImg, YUV_LOWER, YUV_UPPER, binaryImg);

  // p3 = Point(p3.x + 160, p3.y);
  // p4 = Point(p4.x + 160, p4.y);

  // cvtColor(canny_Img, resRGB, CV_GRAY2BGR);
  // if(!left_error){

  //   line(resRGB, p1, p2, COLOR_RED, 4, CV_AA);
  // }
  // if(!right_error){

  //   line(resRGB, p3, p4, COLOR_RED, 4, CV_AA);
  // }
  // // resize(resRGB, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);

  // float left_count = 0;
  // float right_count = 0;

  // if(!right_error)
  // {
  //   float a = (float)(p4.y - p3.y)/(float)(p4.x - p3.x);
  //   float b = (float)(p3.y - a * (float)p3.x) + 10.0;
  //   for(int i = 0 ; i < canny_Img.rows ; i++)
  //   {
  //     for(int j = 0; j < canny_Img.cols ; j++)
  //     {
  //       if(canny_Img.at<uchar>(i, j) == 255 && a * j + b > (float)i) right_count++;
  //     }
  //   }
  // }
  // temp[10] = left_count;
  // temp[11] = right_count;
  // if(right_count > 700)
  // {
  //   return 3;
  // }
  // else return 2;

}


// bool isCaptured = false;

int traffic_light(unsigned char* srcBuf, int iw, int ih, unsigned char* outBuf, int nw, int nh, int centerP[]){

  Mat srcRGB(ih, iw, CV_8UC3, srcBuf); //input
  Mat dstRGB(nh, nw, CV_8UC3, outBuf); //ouput
  Mat resRGB(ih, iw, CV_8UC3);         //result



  Mat roiImg, hsv, redBinaryImg, redBinaryImg1, yellowBinaryImg, greenBinaryImg;
  Mat result;

  roiImg = srcRGB(Rect(0, 0, srcRGB.cols, srcRGB.rows));

  cvtColor(roiImg, hsv, COLOR_BGR2HSV);

  inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER, greenBinaryImg);

  result = greenBinaryImg;
  cvtColor(result, resRGB, CV_GRAY2BGR);
  int success = Labelling(result) - 1;

  if(success == 0)
    return -1;
  vector<Object> vector;

    for(int i = 0; i < success; i++) {
        vector.push_back(Object(result.cols, result.rows, 0, 0, i + 1));
    }

    if(success < 255) {
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (result.at<uchar>(Point(x, y)) > 0) {
                    if (vector[result.at<uchar>(Point(x, y)) - 1].minx > x) {
                        vector[result.at<uchar>(Point(x, y)) - 1].minx = x;
                    }
                    if (vector[result.at<uchar>(Point(x, y)) - 1].miny > y) {
                        vector[result.at<uchar>(Point(x, y)) - 1].miny = y;
                    }
                    if (vector[result.at<uchar>(Point(x, y)) - 1].maxx < x) {
                        vector[result.at<uchar>(Point(x, y)) - 1].maxx = x;
                    }
                    if (vector[result.at<uchar>(Point(x, y)) - 1].maxy < y) {
                        vector[result.at<uchar>(Point(x, y)) - 1].maxy = y;
                    }
                }
            }
        }
    }

    int max_size = 0;
    int max_num = 0;
    for(int i = 0; i < vector.size(); i++) {
        vector[i].width = vector[i].maxx - vector[i].minx;
        vector[i].height = vector[i].maxy - vector[i].miny;
        if(vector[i].width * vector[i].height > max_size) {
            max_size = vector[i].width * vector[i].height;
            max_num = i;
        }
    }

    if(max_size < 100)
      return -1;
    int center_x, center_y, radius;

    center_x = vector[max_num].minx + vector[max_num].width/2;
    center_y = vector[max_num].miny + vector[max_num].height/2;
    radius = center_x - vector[max_num].minx;

    int white = 0;
    int black = 0;
    int maxx = vector[max_num].maxx;
    int maxy = vector[max_num].maxy;
    int minx = vector[max_num].minx;
    int miny = vector[max_num].miny;
    MSG("%d",abs((maxx - minx) - (maxy - miny)));
    if(abs((maxx - minx) - (maxy - miny)) > 10){
      return - 1;
    }
    for(int y = vector[max_num].miny; y < vector[max_num].maxy; y++) {
        for(int x = vector[max_num].minx; x < vector[max_num].maxx; x++) {
            if( (x - center_x)*(x-center_x) + (y-center_y)*(y-center_y) <= radius*radius) {
                if(result.at<uchar>(Point(x, y)) != 0)
                    white++;
                else
                    black++;
            }
        }
    }
    circle(srcRGB, Point(center_x, center_y), radius, Scalar(255, 255, 255), 2);

    if(white+black > 0)
    {
    MSG("white : %d black : %d percent : %f" , white, black, (double)white/(double)(white+black));
    if((double)white/(double)(white+black) > 0.75) {
          return 2;
          putText( srcRGB, "right", Point(10, 40), 2, 1.2, Scalar::all(255) );
    }
    else {
      return 1;
      putText( srcRGB, "left", Point(10, 40), 2, 1.2, Scalar::all(255) );
    }
    // resize(srcRGB, dstRGB, Size(nw, nh), 0, 0, CV_INTER_LINEAR);
} 
  return -1;
}
}

////////////////////밑에 함수들은 exam_cv.cpp에서만 사용하는 함수들/////////////////////

int Labelling(Mat &img) {
    stack <Point> st;

    int number = 1;

    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            if(img.at<uchar>(Point(x, y)) == 255) {
                st.push(Point(x, y));
                while(!st.empty()) {
                    int ky = st.top().y;
                    int kx = st.top().x;
                    st.pop();
                    img.at<uchar>(Point(kx, ky)) = number;
                    if(ky + 1 < img.rows && img.at<uchar>(Point(kx, ky+1)) == 255) {
                        st.push(Point(kx, ky + 1));
                    }
                    if(kx - 1 >= 0 && img.at<uchar>(Point(kx-1, ky)) == 255) {
                        st.push(Point(kx - 1, ky));
                    }
                    if(ky - 1 >= 0 && img.at<uchar>(Point(kx, ky - 1)) == 255) {
                        st.push(Point(kx, ky-1));
                    }
                    if(kx + 1 < img.cols && img.at<uchar>(Point(kx +1, ky)) == 255) {
                        st.push(Point(kx + 1, ky));
                    }
                }
                number ++;
            }
        }
    }
    return number;
}

float get_slope(const Point& p1, const Point& p2) {

	float slope;

	if (p2.y - p1.y != 0.0) {
		slope = ((float)p2.y - (float)p1.y) / ((float)p2.x - (float)p1.x);
	}
	return slope;
}



bool get_intersectpoint(const Point& AP1, const Point& AP2,
	const Point& BP1, const Point& BP2, Point* IP)
{
	double t;
	double s;
	double under = (BP2.y - BP1.y)*(AP2.x - AP1.x) - (BP2.x - BP1.x)*(AP2.y - AP1.y);
	if (under == 0) return false;

	double _t = (BP2.x - BP1.x)*(AP1.y - BP1.y) - (BP2.y - BP1.y)*(AP1.x - BP1.x);
	double _s = (AP2.x - AP1.x)*(AP1.y - BP1.y) - (AP2.y - AP1.y)*(AP1.x - BP1.x);

	t = _t / under;
	s = _s / under;

	if (t<0.0 || t>1.0 || s<0.0 || s>1.0) return false;
	if (_t == 0 && _s == 0) return false;

	IP->x = AP1.x + t * (double)(AP2.x - AP1.x);
	IP->y = AP1.y + t * (double)(AP2.y - AP1.y);

	return true;
}

bool hough_left(Mat& img, Mat& srcRGB, Point* p1, Point* p2) {

	vector<Vec2f> linesL;
  vector<Vec2f> newLinesL;

  Point point1;
  Point point2;

  int count = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  int threshold = 40;

  for (int i = 10; i > 0; i--){

    HoughLines(img, linesL, 1, CV_PI / 180, threshold);

    for(size_t j = 0; j < linesL.size(); j++){

      Vec2f temp;

      float rho = linesL[j][0];
      float theta = linesL[j][1];

      if(CV_PI / 18 >= theta || theta >= CV_PI / 18 * 8) continue;

      temp[0] = rho;
      temp[1] = theta;

      newLinesL.push_back(temp);

    }


    int clusterCount = 2;
    Mat h_points = Mat(newLinesL.size(), 1, CV_32FC2);
    Mat labels, centers;
    if (newLinesL.size() > 1) {
     for (size_t i = 0; i < newLinesL.size(); i++) {
      count++;
      float rho = newLinesL[i][0];
      float theta = newLinesL[i][1];


      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      h_points.at<Point2f>(i, 0) = Point2f(rho, (float)(theta * 100));
    }
    kmeans(h_points, clusterCount, labels,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
      3, KMEANS_RANDOM_CENTERS, centers);

    Point mypt1 = centers.at<Point2f>(0, 0);

    float rho = mypt1.x;
    float theta = (float)mypt1.y / 100;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;

    int _x1 = int(x0 + 1000 * (-b));
    int _y1 = int(y0 + 1000 * (a));
    int _x2 = int(x0 - 1000 * (-b));
    int _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    Point mypt2 = centers.at<Point2f>(1, 0);

    rho = mypt2.x;
    theta = (float)mypt2.y / 100;
    a = cos(theta), b = sin(theta);
    x0 = a * rho, y0 = b * rho;

    _x1 = int(x0 + 1000 * (-b));
    _y1 = int(y0 + 1000 * (a));
    _x2 = int(x0 - 1000 * (-b));
    _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    break;
  };
}
if (count != 0) {
  p1->x = x1 / 2; p1->y = y1 / 2;
  p2->x = x2 / 2; p2->y = y2 / 2;


  return false;
}
return true;
}

bool hough_right(Mat& img, Mat& srcRGB, Point* p1, Point* p2) {

	vector<Vec2f> linesR;
  vector<Vec2f> newLinesR;

  Point point1;
  Point point2;

  int count = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  int threshold = 40;

  for (int i = 10; i > 0; i--){
    HoughLines(img, linesR, 1, CV_PI / 180, threshold);



    for(size_t j = 0; j < linesR.size(); j++){

      Vec2f temp;

      float rho = linesR[j][0];
      float theta = linesR[j][1];

      if(CV_PI / 18 * 10 >= theta || theta >= CV_PI / 18 * 17) continue;

      temp[0] = rho;
      temp[1] = theta;

      newLinesR.push_back(temp);

    }


    int clusterCount = 2;
    Mat h_points = Mat(newLinesR.size(), 1, CV_32FC2);
    Mat labels, centers;
    if (newLinesR.size() > 1) {
     for (size_t i = 0; i < newLinesR.size(); i++) {
      count++;
      float rho = newLinesR[i][0];
      float theta = newLinesR[i][1];


      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      h_points.at<Point2f>(i, 0) = Point2f(rho, (float)(theta * 100));
    }
    kmeans(h_points, clusterCount, labels,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
      3, KMEANS_RANDOM_CENTERS, centers);

    Point mypt1 = centers.at<Point2f>(0, 0);

    float rho = mypt1.x;
    float theta = (float)mypt1.y / 100;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;

    int _x1 = int(x0 + 1000 * (-b));
    int _y1 = int(y0 + 1000 * (a));
    int _x2 = int(x0 - 1000 * (-b));
    int _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    Point mypt2 = centers.at<Point2f>(1, 0);

    rho = mypt2.x;
    theta = (float)mypt2.y / 100;
    a = cos(theta), b = sin(theta);
    x0 = a * rho, y0 = b * rho;

    _x1 = int(x0 + 1000 * (-b));
    _y1 = int(y0 + 1000 * (a));
    _x2 = int(x0 - 1000 * (-b));
    _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    break;
  };
}
if (count != 0) {
  p1->x = x1 / 2; p1->y = y1 / 2;
  p2->x = x2 / 2; p2->y = y2 / 2;

  return false;
}
return true;
}

bool hough_curve(Mat& img, Mat& srcRGB, Point* p1, Point* p2) {

	vector<Vec2f> lines;

  Point point1;
  Point point2;

  int count = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  int threshold = 40;

  for (int i = 10; i > 0; i--){
    HoughLines(img, lines, 1, CV_PI / 180, threshold);


    int clusterCount = 2;
    Mat h_points = Mat(lines.size(), 1, CV_32FC2);
    Mat labels, centers;
    if (lines.size() > 1) {
     for (size_t i = 0; i < lines.size(); i++) {
      count++;
      float rho = lines[i][0];
      float theta = lines[i][1];
      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      h_points.at<Point2f>(i, 0) = Point2f(rho, (float)(theta * 100));
    }
    kmeans(h_points, clusterCount, labels,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
      3, KMEANS_RANDOM_CENTERS, centers);

    Point mypt1 = centers.at<Point2f>(0, 0);

    float rho = mypt1.x;
    float theta = (float)mypt1.y / 100;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    int _x1 = int(x0 + 1000 * (-b));
    int _y1 = int(y0 + 1000 * (a));
    int _x2 = int(x0 - 1000 * (-b));
    int _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    Point mypt2 = centers.at<Point2f>(1, 0);

    rho = mypt2.x;
    theta = (float)mypt2.y / 100;
    a = cos(theta), b = sin(theta);
    x0 = a * rho, y0 = b * rho;

    _x1 = int(x0 + 1000 * (-b));
    _y1 = int(y0 + 1000 * (a));
    _x2 = int(x0 - 1000 * (-b));
    _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    break;
  };
}
if (count != 0) {
  p1->x = x1 / 2; p1->y = y1 / 2;
  p2->x = x2 / 2; p2->y = y2 / 2;

  return false;
}
return true;
}

int curve_detector(Mat& leftImg, Mat& rightImg, int number){

  bool error = true;

  float xLeft, xRight, slope, steer, skewness, y;
  int angle;
  float skewnessValue;

  y = 60.0;

  Mat oriImg, roiImg, hsvImg, binaryImg, binaryImg1, cannyImg;
  Point p1, p2;

  hconcat(leftImg, rightImg, cannyImg);

  switch(number){
    case 1 : case 2 : case 5:
    error = hough_curve(cannyImg, roiImg, &p1, &p2);
    break;
    case 3 : 
    error = hough_left(cannyImg, roiImg, &p1, &p2);
    break;
    case 4 :
    error = hough_right(cannyImg, roiImg, &p1, &p2);
    break;
  }


  if(number == 2){
    skewnessValue = 2.5;
  }
  else{
    skewnessValue = 2.0;
  }

  slope = get_slope(p1, p2);

  if(error){
    return 1500;
  }
    else if(slope < 0.0){ // right rotate

      if(slope < -1.2) slope = -1.2;
      else if(slope > -0.2) slope = -0.2;

      steer =  data_transform(slope, -1.2, -0.2, 100.0, 500.0);

      xLeft = (y - p1.y + slope * p1.x) / slope;

      if(xLeft < -120.0) xLeft = -120.0;
      else if(xLeft > 220.0) xLeft = 220.0;

      skewness = data_transform(xLeft, -120.0, 220.0, 0.0, skewnessValue);

      steer = 1500.0 - (steer * skewness);
      angle = steer;

      if(angle > 1500){
        angle = 1500;
      }

      return angle;
    }
    else{

      if(slope < 0.2) slope = 0.2;
      else if(slope > 1.2) slope = 1.2;

      steer =  data_transform(slope, 0.2, 1.2, -500.0, -100.0);

      xRight = (y - p1.y + slope * p1.x) / slope;

      if(xRight < 100.0) xRight = 100.0;
      else if(xRight > 440.0) xRight = 440.0;

      skewness = data_transform(xRight, 100, 440, -skewnessValue, 0.0);

      steer = 1500.0 + (steer * skewness);
      angle = steer;

      if(angle < 1500){
        return 1500;
      }

      return angle;
    }


  }

  bool hough_horizon(Mat& img, Mat& srcRGB, Point* p1, Point* p2) {

   vector<Vec2f> lines;
   vector<Vec2f> newLines;

   Point point1;
   Point point2;

   int count = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
   int threshold = 40;

   for (int i = 10; i > 0; i--){
    HoughLines(img, lines, 1, CV_PI / 180, threshold);

    for(size_t j = 0; j < lines.size(); j++){

      Vec2f temp;

      float rho = lines[j][0];
      float theta = lines[j][1];

      // if(CV_PI / 18 * 2 <= theta && theta <= CV_PI / 18 * 16) continue;

      temp[0] = rho;
      temp[1] = theta;

      newLines.push_back(temp);

    }


    int clusterCount = 2;
    Mat h_points = Mat(newLines.size(), 1, CV_32FC2);
    Mat labels, centers;
    if (newLines.size() > 1) {
     for (size_t i = 0; i < newLines.size(); i++) {
      count++;
      float rho = newLines[i][0];
      float theta = newLines[i][1];


      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      h_points.at<Point2f>(i, 0) = Point2f(rho, (float)(theta * 100));
    }
    kmeans(h_points, clusterCount, labels,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
      3, KMEANS_RANDOM_CENTERS, centers);

    Point mypt1 = centers.at<Point2f>(0, 0);

    float rho = mypt1.x;
    float theta = (float)mypt1.y / 100;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;

    int _x1 = int(x0 + 1000 * (-b));
    int _y1 = int(y0 + 1000 * (a));
    int _x2 = int(x0 - 1000 * (-b));
    int _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    Point mypt2 = centers.at<Point2f>(1, 0);

    rho = mypt2.x;
    theta = (float)mypt2.y / 100;
    a = cos(theta), b = sin(theta);
    x0 = a * rho, y0 = b * rho;

    _x1 = int(x0 + 1000 * (-b));
    _y1 = int(y0 + 1000 * (a));
    _x2 = int(x0 - 1000 * (-b));
    _y2 = int(y0 - 1000 * (a));

    x1 += _x1;
    y1 += _y1;

    x2 += _x2;
    y2 += _y2;

    break;
  };
}
if (count != 0) {
  p1->x = x1 / 2; p1->y = y1 / 2;
  p2->x = x2 / 2; p2->y = y2 / 2;

  return false;
}
return true;
}

float data_transform(float x, float in_min, float in_max, float out_min, float out_max){
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void get_center_point(Mat& binaryImg, Point * p){

  int cnt_x = 0;
  int cnt_y = 0;
  int count = 0;

  for(int i = 0; i < binaryImg.cols; i++){
    for(int j = 0; j < binaryImg.rows; j++){
      if (binaryImg.at<uchar>(j, i) == 255){
        cnt_x += i;
        cnt_y += j;
        count++;
      }
    }
  }

  if(count > 50){
    p->x = cnt_x / count;
    p->y = cnt_y / count;
  }
}

