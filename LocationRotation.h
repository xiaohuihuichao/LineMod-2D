#ifndef LOCATIONROTATION_H_
#define LOCATIONROTATION_H_

#include <iostream>
#include <math.h>
#include <fstream>
#include <time.h>
#include <thread>

#include"opencv2/opencv.hpp"
#include "config.h"


using namespace cv;
using namespace std;


typedef struct Feature
{
	int m_row, m_col;
	float m_gradVal;
	uchar m_discreteAngle;

	Feature() : m_row(-1), m_col(-1), m_gradVal(-1), m_discreteAngle(0){}
	Feature(int _m_row, int _m_col, float _m_gradVal, uchar _m_discreteAngle) : m_row(_m_row), m_col(_m_col), m_gradVal(_m_gradVal), m_discreteAngle(_m_discreteAngle){}
	Feature(const Feature &f) : m_row(f.m_row), m_col(f.m_col), m_gradVal(f.m_gradVal), m_discreteAngle(f.m_discreteAngle){}
	
	bool operator<(const Feature &f) const
	{
		return m_gradVal < f.m_gradVal;
	}
}Feature;	 //Feature


typedef class Train
{
private:

protected:
	void rotateImg(const Mat &src, Mat &dst, double angle);
	void showImg(char *windowName, const Mat &img);

public:
	unsigned int m_num_feature;
	vector<Feature> features;
	Mat m_templeMat;
	Mat m_rotateTempleMat;
	Mat m_angleMat;
	Mat m_gradMat;
	vector<vector<Feature> > featurePerImg;

	Train(){}
	Train(char *imgPath, int _num_feature = 128);
	Train(const Mat &img, int _num_feature = 128);

	void imageProcess(const int &filterKernel = 7);
	void angleDiscretization();

	void featureSelect(const float &threshold);
	void doTrain(const double &angleBegin = 0, const double &angleEnd = 360, const double &angleStep = 1);

	void operator>>(const char *path) const;
}Train;	//Train


typedef struct Result
{
	Point location;
	float angle;
	float score;

	Result() : location(Point(-1, -1)), angle(-1), score(-1){}
	Result(const int &_x, const int &_y, const float &_angle, const float &_score) : location(Point(_x, _y)), angle(_angle), score(_score){}
	Result(const Point &_p, const double &_angle, const float &_score) : location(Point(_p.x, _p.y)), angle(_angle), score(_score){}
}Result;	//Result


typedef class LocationRotation
{
private:
	Mat img;
	
	Mat m_gradMat;
	float	m_angleStepRough;
	float	m_angleStepPrecise;
	int templeRow;
	int templeCol;


	size_t roughRow;
	size_t roughCol;

	Result m_result[THREAD_NUM];

	Point location;
	float rotation;
	float score;

protected:
	void orOperator(const uchar *src, const int &src_stride, uchar *dst, const int &dst_stride, const int &width, const int &height);
	void imageProcess(const int &filterKernel = 7);
	void angleDiscretization();
	void shiftAngle();
	void spread(const int &t = 3);
	void roughPosition();

public:
	LocationRotation(const Mat &_img);
	LocationRotation(const char *_imgPath);
	LocationRotation() : m_angleStepRough(-1), m_angleStepPrecise(-1), roughRow(-1), roughCol(-1), score(-1), location(Point(-1, -1)), rotation(-1), templeRow(-1), templeCol(-1)
	{
		m_result[THREAD_NUM] = {};
	}

	Mat showMat;
	vector<vector<Feature> > m_features;
	Mat m_angleMat;

	const float& getScore() const
	{
		return score;
	}
	const Point& getLocation() const
	{
		return location;
	}
	const float& getRotation() const
	{
		return rotation;
	}

	const vector<vector<Feature> >& getFeatures() const
	{
		return m_features;
	}

	const Mat& getAngleMat() const
	{
		return m_angleMat;
	}


	void operator<<(const char *path);
	void loadXML(const char *path)
	{
		*this << path;
	}
	
	void loadImg(const char *_imgPath);
	void loadImg(const Mat &img);

	
	float computeResponse(const uchar &modelOri, const uchar &testOris);

	int getRoughRow() const
	{
		return roughRow;
	}

	int getRoughCol() const
	{
		return roughCol;
	}

	void doRun();

	void toFind(const int &angleStart, const int &angleEnd, const int &angleStep, const int &thread_id);

}LocationRotation;	//LocationRotation


void showProgressBar(float total, float current, int num=10);

#endif	//LOCATIONROTATION_H_
