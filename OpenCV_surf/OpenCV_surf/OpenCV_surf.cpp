// OpenCV_surf.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"


//-----------------------------------��ͷ�ļ��������֡�---------------------------------------  
//      ����������������������ͷ�ļ�  
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


//-----------------------------------�������ռ��������֡�--------------------------------------  
//      ����������������ʹ�õ������ռ�  
//-----------------------------------------------------------------------------------------------  
using namespace cv;
using namespace std;
//-----------------------------------���궨�岿�֡�--------------------------------------------   
//      ����������һЩ������ 
//------------------------------------------------------------------------------------------------   
#define WINDOW_NAME1 "��ͼ1��"                  //Ϊ���ڱ��ⶨ��ĺ�   
#define WINDOW_NAME2 "��ͼ2��"        //Ϊ���ڱ��ⶨ��ĺ�   
#define WINDOW_NAME3 "��ֱ����׼���ͼ��"        //Ϊ���ڱ��ⶨ��ĺ�   
#define WINDOW_NAME4 "��ƽ��������׼���ͼ��"        //Ϊ���ڱ��ⶨ��ĺ�   
double g_nangle;
double g_nscale;


//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------  
//      ������ȫ�ֺ���������  
//-----------------------------------------------------------------------------------------------  
static void ShowHelpText();//�����������  
void on_Rotation(int, void *);//�ص����� 


//-----------------------------------��main( )������--------------------------------------------  
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��  
//-----------------------------------------------------------------------------------------------  
int main()
{
	//��0���ı�console������ɫ  
	system("color 3B");

	//��0����ʾ��ӭ�Ͱ�������  
	ShowHelpText();

	//��1�������ز�ͼ  
	Mat srcImage1 = imread("D://1.jpg", 1);
	//Mat srcImage2 = imread("D://2.jpg", 1);
	Mat srcImage2;
	//�����켣��  
	createTrackbar("��ת�Ƕ�", "WINDOW_NAME1", &g_nangle, 1, on_Rotation);

	imshow(WINDOW_NAME1, srcImage1);
	imshow(WINDOW_NAME2, srcImage2);
	Mat dstImage1;
	// ����Ŀ��ͼ��Ĵ�С��������Դͼ��һ��  
	dstImage1 = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	Mat dstImage2;
	// ����Ŀ��ͼ��Ĵ�С��������Դͼ��һ��  
	dstImage2 = Mat::zeros(srcImage2.rows, srcImage2.cols, srcImage2.type());
	Mat dstImage_warp;
	// ����Ŀ��ͼ��Ĵ�С��������Դͼ��һ��  
	dstImage_warp = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	Mat dstImage4;
	// ����Ŀ��ͼ��Ĵ�С��������Դͼ��һ��  
	dstImage4 = Mat::zeros(srcImage1.rows, srcImage1.cols, srcImage1.type());
	if (!srcImage1.data || !srcImage2.data)
	{
		printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false;
	}

	//��2��ʹ��SURF���Ӽ��ؼ���  
	//int minHessian = 700;//SURF�㷨�е�hessian��ֵ  
	static int minHessian;
	cout << "��Ӽ������뺣��������ֵ";
	cin >> minHessian;
	SurfFeatureDetector detector(minHessian);//����һ��SurfFeatureDetector��SURF�� ������������    
	std::vector<KeyPoint> keyPoints1, keyPoints2;//vectorģ���࣬����������͵Ķ�̬����  

	//��3������detect��������SURF�����ؼ��㣬������vector������  
	detector.detect(srcImage1, keyPoints1);
	detector.detect(srcImage2, keyPoints2);
	drawKeypoints(srcImage1, keyPoints1, dstImage1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(srcImage2, keyPoints2, dstImage2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cout << "size of description of Img1: " << keyPoints1.size() << endl;
	cout << "size of description of Img2: " << keyPoints2.size() << endl;

	namedWindow("KeyPoints of image1", 1);
	namedWindow("KeyPoints of image2", 1);

	//����������
	CvFont font;
	double hScale = 1;
	double vScale = 1;
	int lineWidth = 2;// �൱��д�ֵ����� 
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);//��ʼ�����壬׼��д��ͼƬ�ϵ�   
	// cvPoint Ϊ��ʵ�x��y����     
	IplImage* transimg1 = cvCloneImage(&(IplImage)dstImage1);
	IplImage* transimg2 = cvCloneImage(&(IplImage)dstImage2);

	char str1[20], str2[20];
	sprintf(str1, "%d", keyPoints1.size());
	sprintf(str2, "%d", keyPoints2.size());


	const char* str = str1;
	cvPutText(transimg1, str1, cvPoint(280, 230), &font, CV_RGB(255, 0, 0));//��ͼƬ������ַ�   

	str = str2;
	cvPutText(transimg2, str2, cvPoint(280, 230), &font, CV_RGB(255, 0, 0));//��ͼƬ������ַ�   
	//imshow("Description 1",res1);  
	cvShowImage("KeyPoints of image1", transimg1);
	cvShowImage("KeyPoints of image2", transimg2);

	//��4������������������������  
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(srcImage1, keyPoints1, descriptors1);
	extractor.compute(srcImage2, keyPoints2, descriptors2);

	//��5��ʹ��BruteForce����ƥ��  
	// ʵ����һ��ƥ����  
	BruteForceMatcher< L2<float> > matcher;
	std::vector< DMatch > matches;
	//ƥ������ͼ�е������ӣ�descriptors��  
	matcher.match(descriptors1, descriptors2, matches);

	//��6�����ƴ�����ͼ����ƥ����Ĺؼ���  
	Mat imgMatches;
	drawMatches(srcImage1, keyPoints1, srcImage2, keyPoints2, matches, imgMatches);//���л���  
	cout << "number of matched points: " << matches.size() << endl;

	//��7����ʾƥ��Ч��ͼ  
	imshow("ƥ��ͼ", imgMatches);
	//��8��������ƥ�������
	sort(matches.begin(), matches.end()); //����������   
	//��ȡ����ǰN��������ƥ��������  
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<3; i++)
	{
		imagePoints1.push_back(keyPoints1[matches[i].queryIdx].pt);
		imagePoints2.push_back(keyPoints2[matches[i].trainIdx].pt);
	}
	//��9����÷���任  
	Mat warpMat(2, 3, CV_32FC1);
	warpMat = getAffineTransform(imagePoints1, imagePoints2);
	cout << "��õķ���任�����ǣ�\n" << warpMat << endl;
	//��10����Դͼ��Ӧ�øո���õķ���任  
	warpAffine(srcImage1, dstImage_warp, warpMat, dstImage_warp.size());
	imshow(WINDOW_NAME3, dstImage_warp);


	////double adjustValue1 = srcImage1.cols;
	//double adjustValue1 = 400.0;
	////double adjustValue2 = srcImage1.rows;
	//double adjustValue2 = 300.0;
	//Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, adjustValue1, 0, 1.0, adjustValue2, 0, 0, 1.0);
	//cout << "��������Ϊ��\n" << adjustMat << endl << endl;
	//cout << "������任����Ϊ��\n" << adjustMat*warpMat<< endl;
	//warpAffine(srcImage1, dstImage4, warpMat, dstImage4.size());
	//imshow(WINDOW_NAME4, dstImage4);

	waitKey(0);
	return 0;
}

//-----------------------------------��ShowHelpText( )������----------------------------------    
//      ���������һЩ������Ϣ    
//----------------------------------------------------------------------------------------------    
static void ShowHelpText()
{
	//���һЩ������Ϣ    
	printf("\n\n\n\t��ӭ������SURF����������ʾ������~\n\n");
	printf("\t��ǰʹ�õ�OpenCV�汾Ϊ OpenCV CV_VERSION\n\n");
	//printf("\t��Ӽ�������SURF�㷨�е�hessian��ֵ��\n\n");
	printf("\t��������˳�\n\n\n\n\t\t\t\t\t\t\t\t by����\n\n\n");
}
void on_Rotation(int, void *)
{
	Mat getRotationMatrix2D(Point2fcenter, g_nangle, g_nscale);
}