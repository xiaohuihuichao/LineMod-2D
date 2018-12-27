#include "LocationRotation.h"


Train::Train(char *imgPath, int _num_feature) : m_num_feature(_num_feature)
{
	features.clear();
	m_templeMat = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	if (0 == m_templeMat.data)
		{
			cout << "Error in Train::Train(char *imgPath, int _num_feature) : read img failed.\n";
		}
	m_templeMat.copyTo(m_rotateTempleMat);
}


Train::Train(const Mat &img, int _num_feature) : m_num_feature(_num_feature)
{
	features.clear();
	if (0 == img.data)
	{
		cout << "Error in Train::Train(const Mat &img, int _num_feature) : the input img is NULL.\n";
	}
	img.copyTo(m_templeMat);
	m_templeMat.copyTo(m_rotateTempleMat);
}


void Train::imageProcess(const int &filterKernel)
{
	Mat smoothed;
	GaussianBlur(m_rotateTempleMat, smoothed, Size(filterKernel, filterKernel), 0, 0, BORDER_REPLICATE);

	if (1 == m_rotateTempleMat.channels())
	{
		Mat sobel_dx, sobel_dy;
		Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		//CV_32F == float
		//CV_64F == double
		m_gradMat = Mat::zeros(sobel_dx.size(), CV_32F);
		m_gradMat = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);

		phase(sobel_dx, sobel_dy, m_angleMat, true);
	}
}


//将m_angleMat离散为0~8
void Train::angleDiscretization()
{
	m_angleMat = m_angleMat * 16 / 360;
	m_angleMat.convertTo(m_angleMat, CV_8U);

	for (int r = 0; r < m_angleMat.rows; ++r)
	{
		uchar *angle_r = m_angleMat.ptr<uchar>(r);
		for (int c = 0; c < m_angleMat.cols; ++c)
		{
			angle_r[c] &= 7;
		}
	}
}


//特征点筛选
void Train::featureSelect(const float &threshold)
{
	Mat angle_unfilter;
	m_angleMat.copyTo(angle_unfilter);

	vector<Feature> candidates;
	candidates.clear();

	//Roi * Roi邻域
	int Roi = SELECT_ROI;
	for (int r = Roi / 2; r < angle_unfilter.rows - Roi / 2; ++r)
	{
		float *grad_r = m_gradMat.ptr<float>(r);

		for (int c = Roi / 2; c < angle_unfilter.cols - Roi / 2; ++c)
		{
			if (grad_r[c] > threshold*threshold)
			{
				int histogram[8] = { 0 };

				uchar *patchnxn_row = &angle_unfilter.ptr<uchar>(r - Roi / 2)[c - Roi / 2];
				for (int roi_r = 0; roi_r < Roi; ++roi_r)
				{
					for (int roi_c = 0; roi_c < Roi; ++roi_c)
					{
						histogram[patchnxn_row[roi_c]]++;
					}
					/*
						setp[0]: 线的数据量大小，单位为字节	channel * col * sizeof(type)
						setp[1]: 点的数据量大小，单位为字节	channel * sizeof(type)
						step1(0): 线的通道数量，channel * col
						step1(1): 点的通道数量，channel
					*/
					patchnxn_row += angle_unfilter.step1(0);
				}

				int max_votes = -1;
				int index = -1;
				for (int i = 0; i < 8; ++i)
				{
					if (max_votes < histogram[i])
					{
						index = i;
						max_votes = histogram[i];
					}
				}

				if (max_votes > Roi*Roi / 2
					&& angle_unfilter.ptr<uchar>(r)[c] == index)
				{
					Feature f(r, c, grad_r[c], index);
					candidates.emplace_back(f);
				}
			}
		}
	}//在幅度值>threshold 且3x3邻域内某一角度数量超过5则emplace_back()
	if (candidates.size() < m_num_feature)
	{
		m_num_feature = candidates.size();
		cout << "after angle select : num of candidates is " << candidates.size() << ".\n";
	}

	//在上面的feature里面查找其是否为3x3邻域内的极大值
	int nms_kernel_size = SELECT_KERNEL_SIZE;
	auto iter = candidates.begin();
	while (iter != candidates.end())
	{
		int r = iter->m_row;
		int c = iter->m_col;
		float mag = iter->m_gradVal;
		bool top = true;

		for (int r_offset = -nms_kernel_size / 2; (r_offset <= nms_kernel_size / 2) && top; ++r_offset)
		{
			for (int c_offset = -nms_kernel_size / 2; (c_offset <= nms_kernel_size / 2) && top; ++c_offset)
			{
				if ((0 == r_offset && 0 == c_offset)	
					|| r_offset + r < 0 || c_offset + c < 0
					|| r_offset + r >= m_gradMat.rows
					|| c_offset + c >= m_gradMat.cols)
				{
					continue;
				}
				//邻域内有大于特征点的值，则置false
				if (mag < m_gradMat.ptr<float>(r + r_offset)[c + c_offset])
				{
					top = false;
					break;
				}
			}
		}
		if (top)
		{
			++iter;
		}
		else
		{
			iter = candidates.erase(iter);
		}
	}
	if (candidates.size() < m_num_feature)
	{
		m_num_feature = candidates.size();
		cout << "after magnitude select : num of candidates is " << candidates.size() << ".\n";
	}

	//筛选离散的特征点
	sort(candidates.rbegin(), candidates.rend());
	//stable_sort(candidates.rbegin(), candidates.rend());

	//SELECT_DISTANCE
	double distance_sq = 1. * candidates.size() / m_num_feature * 2;

	features.clear();
	//是否满足distance的要求
	for (size_t i = 0; i < candidates.size(); ++i)
	{
		if (features.size() == m_num_feature)
		{
			break;
		}

		bool scattered = true;
		const Feature c = candidates[i];
		for (size_t j = 0; j < features.size() && scattered; ++j)
		{
			const Feature f = features[j];
			scattered = (f.m_row - c.m_row) * (f.m_row - c.m_row)
				+ (f.m_col - c.m_col) * (f.m_col - c.m_col) > distance_sq;
		}
		if (scattered)
		{
			features.emplace_back(c);
		}
	}
	if (features.size() < m_num_feature)
	{
		m_num_feature = features.size();
		cout << "finally, num of features is " << features.size() << ".\n";
	}
}


void Train::showImg(char *windowName, const Mat &img)
{
	namedWindow(windowName, WINDOW_NORMAL);
	imshow(windowName, img);
}


void Train::rotateImg(const Mat &src, Mat &dst, double angle)
{
	Point center = Point(src.cols / 2, src.rows / 2);
	Mat rotMat = getRotationMatrix2D(center, angle, 1.);
	warpAffine(src, dst, rotMat, src.size());
}


void Train::doTrain(const double &angleBegin, const double &angleEnd, const double &angleStep)
{
	cout << "Training:\n";
	featurePerImg.clear();
#ifdef DEBUG
	clock_t start = clock();
#endif // DEBUG

#ifdef TRAIN_IMG_ONE

	for (double angle = angleBegin; angle < angleEnd; angle += angleStep)
	{
		rotateImg(m_templeMat, m_rotateTempleMat, angle);

#ifdef SHOWTRAINPIC
		showImg("AA", m_rotateTempleMat);
		waitKey(0);
#endif // SHOWTRAINPIC
		//	clock_t start = clock();

		imageProcess(GAUSION_FILTER_SIZE);
		angleDiscretization();
		featureSelect(SELECT_THRESHOLD);

		featurePerImg.emplace_back(features);
		showProgressBar(angleEnd, angle, 20);
	}

#else		//TRAIN_IMG_TWO
	for (double angle = angleBegin; angle < angleEnd; angle += angleStep)
	{
		char path[50]={};
		sprintf(path, "%s%d%s", TRAIN_PATH, int(angle), ".bmp");
		
		//cout << path<<"\t";
		m_rotateTempleMat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		m_rotateTempleMat.copyTo(m_templeMat);
		if (0 == m_rotateTempleMat.data)
		{
			cout << "Error in doTrain(): the img path is wrong,\n";
		}

		imageProcess(GAUSION_FILTER_SIZE);
		angleDiscretization();
		featureSelect(SELECT_THRESHOLD);

		featurePerImg.emplace_back(features);
		showProgressBar(angleEnd, angle, 20);
	}

#endif // TRAIN_IMG_TWO

#ifdef DEBUG
	//	cout << "angle_id:" << angle << "\t" << clock() - start << "ms" << endl;
	Mat featureMat = Mat::zeros(m_rotateTempleMat.size(), CV_8U);
	for (size_t i = 0; i < features.size(); ++i)
	{
		featureMat.ptr<uchar>(features[i].m_row)[features[i].m_col] = 255;
	}
#endif // DEBUG
	

#ifdef DEBUG
	cout << "Training time: " << clock() - start <<"ms"<< endl;
#endif // DEBUG
}


void Train::operator>>(const char *path) const
{
	FileStorage fs(path, FileStorage::WRITE);
	
	Feature f;

	fs << "templeRow" << m_templeMat.rows << "templeCol" << m_templeMat.cols;

	for (unsigned int angle = 0; angle < 360; ++angle)
	{
		fs << "angle_" + to_string(angle) << "{";
		//每一度是一个vector<Feature>
		int feature_count = featurePerImg[angle].size();
		fs << "feature_count" << feature_count;
		for (size_t i = 0; i < feature_count; ++i)
		{
			f = featurePerImg[angle][i];
			fs << "feature_" + to_string(i) << "{";
			fs << "m_row" << f.m_row;
			fs	<< "m_col" << f.m_col;
			fs << "m_gradVal" << f.m_gradVal;
			fs	<< "m_discreteAngle" << f.m_discreteAngle;
			fs << "}";
		}
		fs << "}";
	}
	fs.release();
}


LocationRotation::LocationRotation(const Mat &_img) : m_angleStepRough(-1), m_angleStepPrecise(-1), roughRow(-1), roughCol(-1), score(-1), location(Point(-1, -1)), rotation(-1), templeRow(-1), templeCol(-1)
{
	if (0 == _img.data)
	{
		cerr << "Error in LocationRotation(const Mat &_img): _img is empty.\n";
	}
	_img.copyTo(img);
	m_result[THREAD_NUM] = {};
}


LocationRotation::LocationRotation(const char *_imgPath) : m_angleStepRough(-1), m_angleStepPrecise(-1), roughRow(-1), roughCol(-1), score(-1), location(Point(-1, -1)), rotation(-1), templeRow(-1), templeCol(-1)
{
	img = imread(_imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	if (0 == img.data)
	{
		cerr << "Error in LocationRotation(const char *_imgPath):the path is wrong.\n";
	}
	m_result[THREAD_NUM] = {};
}


void LocationRotation::operator<<(const char *path)
{
	FileStorage fs(path, FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Error in LocationRotation::operator<<(const char *path): path is not opend.\n";
		return;
	}

	fs["templeRow"] >> templeRow;
	fs["templeCol"] >> templeCol;

	m_features.clear();
	vector<Feature> vecFeature;
	for (int angle = 0; angle < 360; ++angle)
	{
		FileNode fn = fs["angle_" + to_string(angle)];
		int feature_count = fn["feature_count"];
		vecFeature.clear();
		for (int i = 0; i < feature_count; ++i)
		{
			Feature f;
			FileNode fFn = fn["feature_"+to_string(i)];
			fFn["m_row"] >> f.m_row;
			fFn["m_col"] >> f.m_col;
			fFn["m_gradVal"] >> f.m_gradVal;
			fFn["m_discreteAngle"] >> f.m_discreteAngle;
			
			vecFeature.emplace_back(f);
		}
		m_features.emplace_back(vecFeature);
	}
	fs.release();
}


void LocationRotation::loadImg(const char *_imgPath)
{
	img = imread(_imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	if (0 == img.data)
	{
		cerr << "Error in LocationRotation(const char *_imgPath):the path is wrong.\n";
	}
}


void LocationRotation::loadImg(const Mat &_img)
{
	if (0 == _img.data)
	{
		cerr << "Error in LocationRotation(const Mat &_img): _img is empty.\n";
	}
	_img.copyTo(img);
}


void LocationRotation::roughPosition()
{
	double x_mean = 0, y_mean = 0;
	int num_valid = 0;
	Mat cannyI;

	Canny(img, cannyI, ROUGH_POSITION_LOW_THR, ROUGH_POSITION_HIGH_THR, 3);

	for (int r = 0; r < m_gradMat.rows; ++r)
	{
		uchar * const pR = cannyI.ptr<uchar>(r);
		for (int c = 0; c < cannyI.cols; ++c)
		{
			if (pR[c]>0)
			{
				x_mean += c;
				y_mean += r;
				++num_valid;
			}
		}
	}
	x_mean /= (num_valid + 1);
	y_mean /= (num_valid + 1);

	roughRow = int(y_mean);
	roughCol = int(x_mean);
	
	circle(showMat, Point(roughCol, roughRow), 10, Scalar(0, 255, 0), 5);

//	namedWindow("rough", WINDOW_NORMAL);
//	circle(cannyI, Point(roughCol, roughRow), 6, Scalar(255), 3);
	//imshow("rough", cannyI);
	//waitKey(0);
}


void LocationRotation::imageProcess(const int &filterKernel)
{
	Mat smoothed;
	// For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
	//高斯滤波
	GaussianBlur(img, smoothed, Size(filterKernel, filterKernel), 0, 0, BORDER_REPLICATE);

	if (1 == img.channels())
	{
		Mat sobel_dx, sobel_dy;
		//////////《OpenCV3编程入门》书里说，当kernel_size=3时，使用Scharr其结果更精确，且运行速度与3x3的Sobel一样快。
		//边界模式是BORDER_REPLICATE——复制最边缘的像素。
		Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		//	Scharr(smoothed, sobel_dx, CV_32F, 1, 0, 1.0, 0.0, BORDER_REPLICATE);
		//	Scharr(smoothed, sobel_dx, CV_32F, 0, 1, 1.0, 0.0, BORDER_REPLICATE);
		//幅值的平方
		//CV_32F == float
		//CV_64F == double
		m_gradMat = Mat::zeros(sobel_dx.size(), CV_32F);
		m_gradMat = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
		//根据sobel_x和sobel_y求角度，输出为sobel_ag
		phase(sobel_dx, sobel_dy, m_angleMat, true);
	}
}


//将m_angleMat离散为0~8
void LocationRotation::angleDiscretization()
{
	//m_angleMat * 16.0 / 360。将m_angleMat离散为16个级别0~15
	m_angleMat = m_angleMat * 16 / 360;
	m_angleMat.convertTo(m_angleMat, CV_8U);

	//0~7	==>	0~7		0~180度变为0~7
	//8~15 ==> 0~7		181~360变为0~7
	for (int r = 0; r < m_angleMat.rows; ++r)
	{
		uchar *angle_r = m_angleMat.ptr<uchar>(r);
		for (int c = 0; c < m_angleMat.cols; ++c)
		{
			angle_r[c] &= 7;
		}
	}
}


void LocationRotation::shiftAngle()
{
	uchar shift = -1;
	for (int r = 0; r < m_angleMat.rows; ++r)
	{
		for (int c = 0; c < m_angleMat.cols; ++c)
		{
			shift = m_angleMat.ptr(r)[c];
			m_angleMat.ptr(r)[c] = 1 << shift;
		}
	}
}


void LocationRotation::orOperator(const uchar *src, const int &src_stride, uchar *dst, const int &dst_stride, const int &width, const int &height)
{
	for (int r = 0; r < height; ++r)
	{
		for (int c = 0; c < width; ++c)
		{
			dst[c] |= src[c];
		}

		// Advance to next row
		src += src_stride;
		dst += dst_stride;
	}
}


void LocationRotation::spread(const int &t)
{
	Mat copyAngle;
	m_angleMat.copyTo(copyAngle);
	for (int r = 0; r < t; ++r)
	{
		int height = m_angleMat.rows - r;
		for (int c = 0; c < t; ++c)
		{
			int width = m_angleMat.cols - c;
			orOperator(&copyAngle.ptr<uchar>(r)[c], copyAngle.step1(0), m_angleMat.ptr(0), m_angleMat.step1(0), width, height);
		}
	}
}


float LocationRotation::computeResponse(const uchar &modelOri, const uchar &testOris)
{
	//15		00001111
	//240	11110000
	uchar lsb_4 = testOris & 15;
	uchar msb_4 = (testOris & 240) >> 4;

	const float *lut_low = RESPONSE_LUT + 32 * modelOri;
	const float *lut_hi = lut_low + 16;
	float similarity = max(lut_low[lsb_4], lut_hi[msb_4]);

	return similarity;
}

void LocationRotation::doRun()
{
	score = -1;
	location.x = -1;
	location.y = -1;
	rotation = -1;
	img.copyTo(showMat);
	cvtColor(showMat, showMat, CV_GRAY2BGR);


	imageProcess();
	angleDiscretization();
	shiftAngle();
	spread(T);
	roughPosition();

//	circle(showMat, Point(roughCol, roughRow), 5, Scalar(255, 255, 255), 5);
	
	
	std::thread t0(&LocationRotation::toFind, this, 360 / THREAD_NUM * 0, 360 / THREAD_NUM * 1, 1, 0);
	std::thread t1(&LocationRotation::toFind, this, 360 / THREAD_NUM * 1, 360 / THREAD_NUM * 2, 1, 1);
	std::thread t2(&LocationRotation::toFind, this, 360 / THREAD_NUM * 2, 360 / THREAD_NUM * 3, 1, 2);
	std::thread t3(&LocationRotation::toFind, this, 360 / THREAD_NUM * 3, 360 / THREAD_NUM * 4, 1, 3);


	t0.join();
	t1.join();
	t2.join();
	t3.join();

	//toFind(0, 360, 1, 0);

	for (int i = 0; i < THREAD_NUM; ++i)
	{
		if (score < m_result[i].score)
		{
			score = m_result[i].score;
			location = Point(m_result[i].location);
			rotation = m_result[i].angle;
		}
	}

#ifdef EARLYSTOP_DEBUG

	cout << "Best result: " << location << ", " << rotation << ", " << score << endl;

#endif
	string loc = "Location: (" + to_string(location.x) + ", " + to_string(location.y) + ")";
	string rot = "Rotation: " + to_string(int(rotation));
	string sco = "Score: " + to_string(score);
	putText(showMat, loc, Point(200, 200), 20, 3, Scalar(0, 0, 255), 2);
	putText(showMat, rot, Point(200, 300), 20, 3, Scalar(0, 0, 255), 2);
	putText(showMat, sco, Point(200, 400), 20, 3, Scalar(0, 0, 255), 2);

	double distance = 100;
	int label1Y = int(location.y + sin(rotation*PI / 180.)*distance * 3.5);
	int label1X = int(location.x - cos(rotation*PI / 180.)*distance * 3.5);
	Point label1(label1X, label1Y);

	int label2X = int(location.x - sin(rotation*PI / 180.)*distance);
	int label2Y = int(location.y - cos(rotation*PI / 180.)*distance);
	Point label2(label2X, label2Y);

	line(showMat, location, label1, Scalar(0, 0, 255), 5);
	line(showMat, location, label2, Scalar(0, 0, 255), 5);
	circle(showMat, getLocation(), 40, Scalar(0, 0, 255), 5);
}


void LocationRotation::toFind(const int &angleStart, const int &angleEnd, const int &angleStep, const int &thread_id)
{
	//cout << angleStart << "~" << angleEnd << endl;
	float similarity = 0;
	
	int best_r = -1;
	int best_c = -1;
	
	float maxScore = -1;
	Point p(-1, -1);
	size_t angle_id = -1;

	int rowStart = max(0, int(roughRow - templeRow / 2 - ROW_ROI / 2));
	int rowEnd = min(img.rows - templeRow, int(roughRow - templeRow / 2 + ROW_ROI / 2));

	int colStart = max(0, int(roughCol - templeCol / 2 - COL_ROI / 2));
	int colEnd = min(int(img.cols - templeCol), int(roughCol - templeCol / 2 + COL_ROI / 2));

	//cout << "row: " << rowStart - rowEnd << "col: " << colStart - colEnd << endl;
	rectangle(showMat, Rect(Point(roughCol - COL_ROI / 2, roughRow - ROW_ROI / 2), Size(COL_ROI, ROW_ROI)), Scalar(255, 255, 255), 3);

#ifdef EARLYSTOP_DEBUG

	float simi = 0;
	int loc_c = -1, loc_r = -1;
	int ang = -1;

#endif

	for (int r = rowStart; r < rowEnd; r += (T / 2 + 1))
	{
		for (int c = colStart; c < colEnd;  c += T / 2 + 1)
		{
			//360次

			for (int i = angleStart; i < angleEnd; i += angleStep)
			{
				similarity = 0;

				for (int j = 0; (j < m_features[i].size()); ++j)
				{
					int featureRow = min(m_features[i][j].m_row + r, m_angleMat.rows - 1);
					int featureCol = min(m_features[i][j].m_col + c, m_angleMat.cols - 1);

					uchar testOris = m_angleMat.ptr<uchar>(featureRow)[featureCol];
					uchar modelOri = m_features[i][j].m_discreteAngle;
					float sim = computeResponse(modelOri, testOris);
					similarity += sim;
					
#ifdef EARLYSTOP_DEBUG


#endif // EARLYSTOP_DEBUG

#ifdef EARLYSTOP
					if (30 == j)
					{
						if (similarity < 30 * (EARLYSTOP_SCALE - 0.2))
						{
							n30++;
							i += 1 * angleStep;
							break;
						}
					}
					else if (60 == j)
					{
						if (similarity < 60 * (EARLYSTOP_SCALE - 0.1))
						{
							n60++;
							i += 1 * angleStep;
							break;
						}
					}
					else if(128 == j)
					{
						if (similarity < 128 * EARLYSTOP_SCALE)
						{
							n128++;
							i += 1 * angleStep;
							break;
						}
					}
					else if (256 == j)
					{
						if (similarity < 256 * EARLYSTOP_SCALE)
						{
							n256++;
							i += 1 * angleStep;
							break;
						}
					}
					else if (FEATURE_NUM / 4 == j)
					{
						if (similarity < FEATURE_NUM / 4 * EARLYSTOP_SCALE)
						{
							n1_4++;
							i += 1 * angleStep;
							break;
						}
					}
					else if (FEATURE_NUM / 3 == j)
					{
						if (similarity < FEATURE_NUM / 3 * EARLYSTOP_SCALE)
						{
							n1_3++;
							i += 1 * angleStep;
							break;
						}
					}
					else if (FEATURE_NUM / 2 == j)
					{
						if (similarity < FEATURE_NUM / 2 * EARLYSTOP_SCALE)
						{
							n1_2++;
							i += 1 * angleStep;
							break;
						}
					}
#endif
				}

				if (similarity > maxScore)
				{
					best_r = r;
					best_c = c;
					angle_id = i;
					p = Point(c + templeCol / 2, r + templeRow / 2);
					maxScore = similarity;
				}
			}

			

#ifdef EARLYSTOP_DEBUG
		//	cout << "(" << loc_c << ", " << loc_r << "): " << simi << ", " << ang << ", " << thread_id << endl;
#endif

		}
	}
//	cout << "time:" << time << "ms\t" << "maxScore:" << maxScore / FEATURE_NUM << "\n" << "angle_id:" << angle_id << "\t" << p << endl;
//	cout << "location distance:" << p - Point(roughCol, roughRow) << endl;
//	cout << "======================================================\n";
	//circle(showMat, Point(best_c, best_r), 10, Scalar(255, 0, 0), 5);
	m_result[thread_id] = Result(p, angle_id, maxScore);

//	putText(showMat, to_string(angle_id), p, 2, 5, Scalar(255, 0, 0));
	circle(showMat, p, 20, Scalar(255, 255, 0), 2);

	double distance = 100;
	int label1Y = int(p.y + sin(angle_id*PI / 180.)*distance * 3.5);
	int label1X = int(p.x - cos(angle_id*PI / 180.)*distance * 3.5);
	Point label1(label1X, label1Y);

	int label2X = int(p.x - sin(angle_id*PI / 180.)*distance);
	int label2Y = int(p.y - cos(angle_id*PI / 180.)*distance);
	Point label2(label2X, label2Y);

	line(showMat, p, label1, Scalar(255, 255, 0), 2);
	line(showMat, p, label2, Scalar(255, 255, 0), 2);
}


void showProgressBar(float total, float current, int num)
{
	float percent = current / total * 100.;
	int n = 100 / num;
	cout << "\t[";
	printf("%.1f", percent);
	cout << "%][";

	for (int i = 0; i < num; ++i)
	{
		if (i < percent / n)
		{
			cout << "=>";
		}
		else
		{
			cout << "  ";
		}
	}
	if (percent<(100 + 100 / (total - 1)) && percent>(100 - 100 / (total - 1)))
	{
		cout << "]\ndone!\n";
	}
	else
	{
		cout << "]" << "\r";
	}
}
