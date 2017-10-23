// OpenCV_surf.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


//-----------------------------------【头文件包含部分】---------------------------------------  
//      描述：包含程序所依赖的头文件  
//----------------------------------------------------------------------------------------------  
#include "opencv2/core/core.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <opencv2/nonfree/nonfree.hpp>  
#include<opencv2/legacy/legacy.hpp>  
#include <iostream>  
#include "highgui.h"  
#include "cv.h"  
#include "vector"  
#include "opencv\cxcore.hpp"  
#include "iostream"  
#include "opencv.hpp"  
#include "opencv2/nonfree/features2d.hpp"
#include <Windows.h>


//-----------------------------------【命名空间声明部分】--------------------------------------  
//      描述：包含程序所使用的命名空间  
//-----------------------------------------------------------------------------------------------  
using namespace cv;
using namespace std;
//-----------------------------------【宏定义部分】--------------------------------------------   
//      描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------   
#define WINDOW_NAME1 "【图1】"                  //为窗口标题定义的宏   
#define WINDOW_NAME2 "【图2】"        //为窗口标题定义的宏   
#define WINDOW_NAME3 "【直接配准后的图】"        //为窗口标题定义的宏   
#define WINDOW_NAME4 "【平移修正配准后的图】"        //为窗口标题定义的宏   


//-----------------------------------【全局函数声明部分】--------------------------------------  
//      描述：全局函数的声明  
//-----------------------------------------------------------------------------------------------  
static void ShowHelpText();//输出帮助文字 
void on_Rotation(int,int,int, void *);//回调函数  
void PSNR_count(Mat, Mat);
//void on_Rotation_scale(int, void *);//回调函数 不是每一个bar要一个回调函数，参考改变亮度对比的
void onMouse(int event, int x, int y, int flags, void* param);
Mat srcImage2, dstImage2,dstImage_warp;
Mat srcImage1, dstImage1;
Mat Rotation_matrix;
Point2f Point2fcenter=float((200.0,150.0));
int g_n_Rotation_angle = 11;
int g_n_Rotation_scale=11;
int g_nminHessian=111;

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat *im = reinterpret_cast<Mat*>(param);
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:     //鼠标左键按下响应：返回坐标和灰度  
		std::cout << "at(" << x << "," << y << ")value is:"
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
	// Point2fcenter = Point(x, y);
		break;
	//case CV_EVENT_RBUTTONDOWN:    //鼠标右键按下响应：输入坐标并返回该坐标的灰度  
	//	std::cout << "input(x,y)" << endl;
	//	std::cout << "x =" << endl;
	//	cin >> x;
	//	std::cout << "y =" << endl;
	//	cin >> y;
	//	std::cout << "at(" << x << "," << y << ")value is:"
	//		<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
	//	break;
	}
}

//-----------------------------------【main( )函数】--------------------------------------------  
//      描述：控制台应用程序的入口函数，我们的程序从这里开始执行  
//-----------------------------------------------------------------------------------------------  
int main()
{
	//【0】改变console字体颜色  
	system("color 3B");

	

	//【0】载入素材图  
     srcImage1 = imread("D://1.jpg", 1);
	 imshow(WINDOW_NAME1, srcImage1);
	 //【0】显示欢迎和帮助文字  
	 ShowHelpText();
	 srcImage2 = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	//Mat srcImage2 = imread("D://2.jpg", 1);
	
	//imshow(WINDOW_NAME2, srcImage2);
	// 设置目标图像的大小和类型与源图像一致  
	
	Mat dstImage1;
	// 设置目标图像的大小和类型与源图像一致  
	dstImage1 = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	Mat dstImage2;
	// 设置目标图像的大小和类型与源图像一致  
	dstImage2 = Mat::zeros(srcImage2.rows, srcImage2.cols, srcImage2.type());
	Mat dstImage_warp;
	// 设置目标图像的大小和类型与源图像一致  
	dstImage_warp = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	Mat dstImage4;
	// 设置目标图像的大小和类型与源图像一致  
	dstImage4 = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	if (!srcImage1.data)
	{
		printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false;
	}

	//创建轨迹条 和获取鼠标左键位置 
	namedWindow(WINDOW_NAME2, 1);
	cv::setMouseCallback(WINDOW_NAME1, onMouse, reinterpret_cast<void*> (&srcImage1));
	createTrackbar("角度", WINDOW_NAME2, &g_n_Rotation_angle, 300, on_Rotation);
	createTrackbar("尺寸", WINDOW_NAME2, &g_n_Rotation_scale, 13, on_Rotation);
	createTrackbar("H阈值", WINDOW_NAME2, &g_nminHessian, 999, on_Rotation);//两个bar调用同一个函数
	//调用回调函数
	on_Rotation(g_n_Rotation_angle, g_n_Rotation_scale, g_nminHessian, 0);
	//on_Rotation(g_n_Rotation_scale, 0);
	//on_Rotation(g_nminHessian, 0);
	

	//【2】使用SURF算子检测关键点  
	//int minHessian = 700;//SURF算法中的hessian阈值  
	/*static int minHessian;
	cout << "请从键盘输入海塞矩阵阈值";
	cin >> minHessian;*/
	


	////double adjustValue1 = srcImage1.cols;
	//double adjustValue1 = 400.0;
	////double adjustValue2 = srcImage1.rows;
	//double adjustValue2 = 300.0;
	//Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, adjustValue1, 0, 1.0, adjustValue2, 0, 0, 1.0);
	//cout << "调整矩阵为：\n" << adjustMat << endl << endl;
	//cout << "调整后变换矩阵为：\n" << adjustMat*warpMat<< endl;
	//warpAffine(srcImage1, dstImage4, warpMat, dstImage4.size());
	//imshow(WINDOW_NAME4, dstImage4);

	waitKey(0);
	return 0;
}

//-----------------------------------【ShowHelpText( )函数】----------------------------------    
//      描述：输出一些帮助信息    
//----------------------------------------------------------------------------------------------    
static void ShowHelpText()
{
	//输出一些帮助信息    
	printf("\n\n\n\t欢迎来到【SURF特征描述】示例程序~\n\n");
	printf("\t当前使用的OpenCV版本为 OpenCV CV_VERSION\n\n");
	//printf("\t请从键盘输入SURF算法中的hessian阈值：\n\n");
	printf("\t按任意键退出\n\n\n\n\t\t\t\t\t\t\t\t by晨光\n\n\n");
}
//-----------------------------【Process( )函数】------------------------------------  
//            描述：进行自定义的旋转、配准操作  
//-----------------------------------------------------------------------------------------  
void Process()
{//根据trackbar的值计算旋转矩阵
	DWORD start_time = GetTickCount();
	double angle = (double)(g_n_Rotation_angle / 10);
	double scale = (double)(g_n_Rotation_scale / 10);
	int minHessian = g_nminHessian;
	Rotation_matrix = getRotationMatrix2D(Point2fcenter, angle, scale);
	cout << "旋转矩阵为：" << Rotation_matrix;
	warpAffine(srcImage1, srcImage2, Rotation_matrix, srcImage2.size());
	//显示效果图 
	imshow(WINDOW_NAME2, srcImage2);
	
	SurfFeatureDetector detector(minHessian);//定义一个SurfFeatureDetector（SURF） 特征检测类对象    
	std::vector<KeyPoint> keyPoints1, keyPoints2;//vector模板类，存放任意类型的动态数组  

	//【3】调用detect函数检测出SURF特征关键点，保存在vector容器中  
	detector.detect(srcImage1, keyPoints1);
	detector.detect(srcImage2, keyPoints2);
	drawKeypoints(srcImage1, keyPoints1, dstImage1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(srcImage2, keyPoints2, dstImage2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cout << "size of description of Img1: " << keyPoints1.size() << endl;
	cout << "size of description of Img2: " << keyPoints2.size() << endl;

	namedWindow("KeyPoints of image1", 1);
	namedWindow("KeyPoints of image2", 1);

	//绘制特征点
	CvFont font;
	double hScale = 1;
	double vScale = 1;
	int lineWidth = 2;// 相当于写字的线条 
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);//初始化字体，准备写到图片上的   
	// cvPoint 为起笔的x，y坐标     
	IplImage* transimg1 = cvCloneImage(&(IplImage)dstImage1);
	IplImage* transimg2 = cvCloneImage(&(IplImage)dstImage2);

	char str1[20], str2[20];
	sprintf(str1, "%d", keyPoints1.size());
	sprintf(str2, "%d", keyPoints2.size());


	const char* str = str1;
	cvPutText(transimg1, str1, cvPoint(280, 230), &font, CV_RGB(255, 0, 0));//在图片中输出字符   

	str = str2;
	cvPutText(transimg2, str2, cvPoint(280, 230), &font, CV_RGB(255, 0, 0));//在图片中输出字符   
	//imshow("Description 1",res1);  
	cvShowImage("KeyPoints of image1", transimg1);
	cvShowImage("KeyPoints of image2", transimg2);

	//【4】计算描述符（特征向量）  
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(srcImage1, keyPoints1, descriptors1);
	extractor.compute(srcImage2, keyPoints2, descriptors2);

	//【5】使用BruteForce进行匹配  
	// 实例化一个匹配器  
	BruteForceMatcher< L2<float> > matcher;
	std::vector< DMatch > matches;
	//匹配两幅图中的描述子（descriptors）  
	matcher.match(descriptors1, descriptors2, matches);

	//【6】绘制从两个图像中匹配出的关键点  
	Mat imgMatches;
	drawMatches(srcImage1, keyPoints1, srcImage2, keyPoints2, matches, imgMatches);//进行绘制  
	cout << "number of matched points: " << matches.size() << endl;

	//【7】显示匹配效果图  
	imshow("匹配图", imgMatches);
	//【8】特征点匹配对排序
	sort(matches.begin(), matches.end()); //特征点排序   
	//获取排在前N个的最优匹配特征点  
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<3; i++)
	{
		imagePoints1.push_back(keyPoints1[matches[i].queryIdx].pt);
		imagePoints2.push_back(keyPoints2[matches[i].trainIdx].pt);
	}
	//【9】求得仿射变换  
	Mat warpMat(2, 3, CV_32FC1);
	warpMat = getAffineTransform(imagePoints1, imagePoints2);
	cout << "求得的仿射变换矩阵是：\n" << warpMat << endl;
	//【10】对源图像应用刚刚求得的仿射变换  
	warpAffine(srcImage1, dstImage_warp, warpMat, dstImage_warp.size());
	imshow(WINDOW_NAME3, dstImage_warp);
	//【11】计算两幅图像 dstImage_warp,srcImage2之间的峰值信噪比
	PSNR_count(dstImage_warp, srcImage2);
	DWORD end_time = GetTickCount();
	cout << "The run time is:" << (end_time - start_time) << "ms!" << endl;
	printf("\t\t\t\t\n\n\n\n");
	

}
//-----------------------------【on_回调( )函数】------------------------------------  
//            描述：实现旋转，缩放变换回调函数  
//------------------------------------------------------------------------------------------------
void on_Rotation(int, void *)//回调函数  
{
	Process();
}
//-----------------------------【计算PSNR函数】------------------------------------  
//            描述：转换为灰度图之后，对比像素差异
//--------------------------------------------------------------------------------------
void PSNR_count(Mat x,Mat y)
{
	double MSE,PSNR0;
	double s = 0;
	int PSNR_dB;
	Mat gray1, gray2;
	
	cvtColor(x, gray1, CV_BGR2GRAY);
	cvtColor(y, gray2, CV_BGR2GRAY);
	for (int i = 0; i < gray1.rows; i++)//不能是小于等于，否则溢出！
	{
		for (int j = 0; j < gray1.cols; j++)
		{
			s = s + int((gray1.at<uchar>(i, j) - gray2.at<uchar>(i, j)) ^ 2);//注意循环顺序和坐标顺序，先进行行循环，对应y坐标，at时放在前面
		}
	}
	MSE = s / ((gray1.rows)*(gray1.cols));
	PSNR0 = (255 ^ 2 )/ MSE;
	PSNR_dB = 10 * log10(PSNR0);
	cout << "复原前后峰值信噪比是：" << PSNR_dB << "dB"<<endl;
	
}