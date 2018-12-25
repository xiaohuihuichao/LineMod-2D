#ifndef CONFIG_H_
#define	CONFIG_H_

#define			EPSINON								1e-14
#define			PI											3.141592654

#define			WATCH
#define			TRAIN									0
//#define			 EARLYSTOP
#define			EARLYSTOP_DEBUG
#define			EARLYSTOP_SCALE				0.94	//0.955

#define			DEBUG
#define			THREAD_NUM						4


#define			T											3
#define			GAUSION_FILTER_SIZE			7	//7	5	3
#define			SELECT_ROI							5		//angle	3 2
#define			SELECT_THRESHOLD			120//120
#define			SELECT_KERNEL_SIZE			3		//mag
#define			ROUGH_POSITION_HIGH_THR		600
#define			ROUGH_POSITION_LOW_THR		550
//#define		SELECT_DISTANCE	
//#define		FEATURE_NUM
//#define		ROW_ROI
//#define		COL_ROI


#define		TRAIN_IMG_ONE		//自己翻转
//#define			TRAIN_IMG_TWO		//读取大图

#ifdef			TRAIN_IMG_ONE

#ifdef WATCH

#define	FEATURE_NUM				1024
#define	TRAIN_PATH					"../watch/tpl1.bmp"
#define	SAVE_PATH					"../watch/back_2_result/"
#define	SAVE_THREAD_PATH	"../watch/back_1_result/"
#define	TEST_PATH					"../watch/back_1/"
#define	NUM							37
#define	XML_PATH					"../watch/back_1/model_d2_1024.xml"
#define	ROW_ROI						2 * 150
#define	COL_ROI						2 * 150

//back_2
//#define	ROW_ROI						2 * 150
//#define	COL_ROI						2 * 250


#endif // WATCH

#ifdef PS4

#define	FEATURE_NUM				256
#define	TRAIN_PATH					"../ps4/temple.bmp"
#define	SAVE_PATH					"../ps4_result/"
#define	TEST_PATH					"../ps4/"
#define	SAVE_THREAD_PATH	"../ps4_thread_result/"
#define	NUM							5
#define	XML_PATH					"../data/ps4_256.xml"
#define	ROW_ROI						2 * 120
#define	COL_ROI						2 * 120

#endif // PS4

#endif // TRAIN_IMG_ONE


#ifdef			TRAIN_IMG_TWO

#ifdef			WATCH

#define			FEATURE_NUM						1024
#define			TRAIN_PATH							"../back_1_2/"
#define			SAVE_THREAD_PATH			"../watch/back_2_2_result/"
#define			TEST_PATH							"../watch/back_2_2/"
#define			NUM									11
#define			XML_PATH							"../model_back2_1024.xml"
#define			ROW_ROI								2 * 200
#define			COL_ROI								2 * 350

#endif // WATCH

#ifdef PS4

#define	FEATURE_NUM				256
#define	TRAIN_PATH					"../ps4/temple.bmp"
#define	SAVE_PATH					"../ps4_result/"
#define	TEST_PATH					"../ps4/"
#define	NUM							5
#define	XML_PATH					"../data/ps4_256.xml"
#define	ROW_ROI						2 * 160
#define	COL_ROI						2 * 160

#endif // PS4

#endif // TRAIN_IMG_TWO


#define		theta0			1.0000		//cos(0)
#define		theta1			0.9239		//cos(22.5)
#define		theta2			0.7071		//cos(45)
#define		theta3			0.3826		//cos(67.5)
#define		theta4			0.0000		//cos(90)

#define		theta5			0.3826
#define		theta6			0.7071
#define		theta7			0.9239

//原始打分
//#define		theta0			1.0000		//cos(0)
//#define		theta1			0.9239		//cos(22.5)
//#define		theta2			0.7071		//cos(45)
//#define		theta3			0.3826		//cos(67.5)
//#define		theta4			0.0000		//cos(90)
//
//#define		theta5			0.3826
//#define		theta6			0.7071
//#define		theta7			0.9239


const float RESPONSE_LUT[256] = {
	0, theta0, theta1, theta0, theta2, theta0, theta1, theta0, theta3, theta0, theta1, theta0, theta2, theta0, theta1, theta0,
	0, theta4, theta5, theta5, theta6, theta6, theta6, theta6, theta7, theta7, theta7, theta7, theta7, theta7, theta7, theta7,
	0, theta1, theta0, theta0, theta1, theta1, theta0, theta0, theta2, theta1, theta0, theta0, theta1, theta1, theta0, theta0,
	0, theta3, theta4, theta3, theta5, theta5, theta5, theta5, theta6, theta6, theta6, theta6, theta6, theta6, theta6, theta6,
	0, theta2, theta1, theta1, theta0, theta0, theta0, theta0, theta1, theta1, theta1, theta1, theta0, theta0, theta0, theta0,
	0, theta2, theta3, theta2, theta4, theta2, theta3, theta2, theta5, theta2, theta5, theta2, theta5, theta2, theta5, theta2,
	0, theta3, theta2, theta2, theta1, theta1, theta1, theta1, theta0, theta0, theta0, theta0, theta0, theta0, theta0, theta0,
	0, theta1, theta2, theta1, theta3, theta1, theta2, theta1, theta4, theta1, theta2, theta1, theta3, theta1, theta2, theta1,
	0, theta4, theta3, theta3, theta2, theta2, theta2, theta2, theta1, theta1, theta1, theta1, theta1, theta1, theta1, theta1,
	0, theta0, theta1, theta0, theta2, theta0, theta1, theta0, theta3, theta0, theta1, theta0, theta2, theta0, theta1, theta0,
	0, theta5, theta4, theta5, theta3, theta3, theta3, theta3, theta2, theta2, theta2, theta2, theta2, theta2, theta2, theta2,
	0, theta1, theta0, theta0, theta1, theta1, theta0, theta0, theta2, theta1, theta0, theta0, theta1, theta1, theta0, theta0,
	0, theta6, theta5, theta6, theta4, theta6, theta5, theta6, theta3, theta6, theta3, theta6, theta3, theta6, theta3, theta6,
	0, theta2, theta1, theta1, theta0, theta0, theta0, theta0, theta1, theta1, theta1, theta1, theta0, theta0, theta0, theta0,
	0, theta7, theta6, theta7, theta5, theta7, theta6, theta7, theta4, theta7, theta6, theta7, theta5, theta7, theta6, theta7,
	0, theta3, theta2, theta2, theta1, theta1, theta1, theta1, theta0, theta0, theta0, theta0, theta0, theta0, theta0, theta0
};


#endif // !CONFIG_H_
