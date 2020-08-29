#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

struct Object {
	int id;
	int life;
	int x, y, w, h;
	int grade;
	Scalar color;
	bool isCount;
	Point center;
	vector<Rect> bound;
	vector<Point> tjOrigin;
	vector<Point> tjPredict;
};
vector<Object> objs;


Mat erosion_dst, dilation_dst;
Mat src, src_gray, dst;
Mat grad;
Mat elementErosion;
Mat elementDilation;
Mat frame;
Mat background;
Mat object;
Mat src3_gray;
Mat frameorg;

const char* window_name = "Sobel Demo - Simple Edge Detector";
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

int	objIdx =-1;
int delCount = 0;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 24;
int const max_elem = 4;
int const max_kernel_size = 40;

// Function Headers 
void Erosion(int, void*);
void Dilation(int, void*);
void thresh_callback(int, void*);

void RemoveLife(int i);
void IncreaseLife(int i);
void Release(void);
void AddExistObject(int objIdx, Rect boundRect);
void AddNewObject(Rect boundRect, int i, Scalar color);
void UpdateObject(Rect bound, int i);
void UpdateObjectLife();
bool isZeroLife(int i);
int ExistObject(vector<Rect> bound);
double CalDistance(Rect boundRect, Object objs, int i);

int main(int argc, char** argv)
{
	//namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
	//namedWindow("Dilation Demo", WINDOW_AUTOSIZE);
	//moveWindow("Dilation Demo", frame.cols, 0);

	VideoCapture cap("C:\\Users\\neaynie\\source\\repos\\LAB1\\video\\video.avi");
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	cap.read(frame);
	Mat acc = Mat::zeros(frame.size(), CV_32FC3);
	for (;;)
	{
		cap.read(frame);

		Mat gray;
		cap >> frame; // get a new frame from camera
		cap >> frameorg;
		//imshow("Original", frame);

		GaussianBlur(frame, frame, Size(25, 25), 0, 0);

		// Get 50% of the new frame and add it to 50% of the accumulator
		accumulateWeighted(frame, acc, 0.5);
		// Scale it to 8-bit unsigned
		convertScaleAbs(acc, background);

		//imshow("Weighted Average", background);
		//blur(background, background, Size(3, 3));

		subtract(background, frame, frame);
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		//imshow("Gray", frame);
		threshold(frame, frame, 15, 255, THRESH_BINARY);

		//imshow("threshold", frame);


		/// Generate grad_x and grad_y 
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(frame, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		imwrite("sobel_x.jpg", abs_grad_x);

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(frame, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		imwrite("sobel_y.jpg", abs_grad_y);

		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
		//imwrite("sobel.jpg", grad);
		//imshow("frame", frame);
		//imshow("src3_gray", src3_gray);

		//imshow("grad", grad);



		/// Create Erosion Trackbar
		//createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo", &erosion_elem, max_elem, Erosion);
		//createTrackbar("Kernel size:\n 2n +1", "Erosion Demo", &erosion_size, max_kernel_size, Erosion);

		/// Create Dilation Trackbar
		//createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo", &dilation_elem, max_elem, Dilation);
		//createTrackbar("Kernel size:\n 2n +1", "Dilation Demo", &dilation_size, max_kernel_size, Dilation);

		/// Default start
		Dilation(0, 0);
		Erosion(0, 0);


		thresh_callback(0, 0);
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(erosion_dst, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours 
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		//rectangle(frameorg, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/////// AJ mark //////

	cout << objs.size();
	cout << "    ";

	for (int i = 0; i < boundRect.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		objIdx = ExistObject(boundRect);

		if(objs.size() == 0)
			AddNewObject(boundRect[i], i, color);

		else if (objIdx == -1) {  //not exist
			//cout << "  NewObj";
			AddNewObject(boundRect[i], i, color);
		}
		else { //exist
			//cout << "update"<< i;
			UpdateObject(boundRect[i], i);
			//cout << "End";
		}

		UpdateObjectLife();

		

	}

	for (int i = 0; i < objs.size(); i++)
		{
			rectangle(frameorg, objs[i].bound[objs[i].bound.size()-1].tl(), objs[i].bound[objs[i].bound.size() - 1].br(), objs[i].color, 2, 8, 0);
		}
	/// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", frameorg);
}


/*
void AddExistObject(int objIdx, Rect boundRect) {
	UpdateObjectLife();
}
*/

void AddNewObject(Rect boundRect, int i, Scalar color) {
	Object obj;
	Point center;
	obj.color = color;
	obj.life = 2;
	//obj.center.x = boundRect.x;
	//obj.center.y = boundRect.y;
	obj.bound.push_back(boundRect);

	objs.push_back(obj);
	//cout << boundRect.x;
	//cout << "  ";
	//cout << "DebugAdd";
	//cout << objs.size();

	//if (objs.size() > 20000) objs.id = 0;
	//UpdateObjectLife();

}


void UpdateObjectLife() {
	for (int i = 0; i < objs.size(); i++) {
		//objs.erase(std::remove_if(objs.begin(), objs.end(),isZeroLife(i)),objs.end());
		if (isZeroLife(i)) {
			//cout << "remove:" << objs.size();
			objs.erase(objs.begin()+i);
			//delCount++;
		}
		else
		//cout << "  begore : ";
		//cout << objs[i].life;
			RemoveLife(i);
		//cout << " after :";
		//cout << objs[i].life;
	}
}

bool isZeroLife(int i) {
	if (objs[i].life <= 0)
		return true;
	return false;
}
void RemoveLife(int i)
{
	objs[i].life--;
}

void IncreaseLife(int i)
{
	objs[i].life = 2;
}



void UpdateObject(Rect bound, int i) {
	//cout << delCount;
	objs[objs.size()-1].bound.push_back(bound);
	IncreaseLife(objs.size() - 1);
	//objs[i].tjOrigin.push_back(Point(bound.x, bound.y));
	//if ((int)objs[i].tjOrigin.size() > 5)
		//objs[i].tjOrigin.erase(objs[i].tjOrigin.begin(), objs[i].tjOrigin.begin() + 1);
	// IncreaseLife(i);
}





int ExistObject(vector<Rect> bound) {
	double minDist = 10000, dist;
	

	for (int i = 0; i < objs.size(); i++) {
		//cout << "DebugExist";
		dist = CalDistance(bound[i], objs[i], i);
		if (dist < minDist) {
			//cout << "Debug   ";
			minDist = dist;
			objIdx = i;
		}
		
		if (minDist > 380) {
			//cout << "    NewObj";
			//cout << objs.size();
			objIdx = -1;
		}
		//cout << "   mindist:";
		//cout << "mindist :"<< minDist;
		return objIdx;
	}
}


double CalDistance(Rect boundRect, Object objs, int i) {
	return sqrt(pow(boundRect.x - objs.bound[i].x, 2) + pow(boundRect.y - objs.bound[i].y, 2));
}




/**  @function Erosion  */
void Erosion(int, void*)
{
	int erosion_type;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	elementErosion = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	erode(dilation_dst, erosion_dst, elementErosion);
	dilate(erosion_dst, erosion_dst, elementDilation);

	//imshow("Erosion Demo", erosion_dst);
}

/** @function Dilation */
void Dilation(int, void*)
{
	int dilation_type;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	elementDilation = getStructuringElement(dilation_type, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));

	/// Apply the dilation operation
	dilate(frame, dilation_dst, elementDilation);
	erode(dilation_dst, dilation_dst, elementErosion);
	//imshow("Dilation Demo", dilation_dst);
}




