
#ifndef __BINARIZE_UTILS_H__
#define __BINARIZE_UTILS_H__

//#include <opencv2/core/core.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

using namespace std;
using namespace cv;

const int DEFAULT_WINDOW_SIZE = 15;
const int BERNSEN_WINDOW = 31;
const float BERNSEN_K = 0.2;
const int NIBLACK_WINDOW = 15;
const float NIBLACK_K = -0.2;
const int SAUVOLA_WINDOW = 25;
const float SAUVOLA_K = 0.5;
const int WOLF_WINDOW = 15;
const float WOLF_K = 0.5;
const int NICK_WINDOW = 75;
const float NICK_K = -0.2;
const float GATOS_K = 0.2, GATOS_GLYPH = 60;
const int SINGH_WINDOW = 75;
const float SINGH_K = 0.2;
const float WAN_K = 0.2;
const float ISAUVOLA_K = 0.01;

enum AlgCode {
    Otsu = 0,
    Bernsen = 1,
    Niblack = 2,
    Sauvola = 3,
    Wolf = 4,
    Nick = 5,
    GatosBeta = 6,
    SuLu = 7,
    Singh = 8,
    Bataineh = 9,
    WAN = 10,
    ISauvola = 11,
    JiaShi = 12,
};

bool binarizeImage(Mat imageIN, string fnameOUT, int algCode, int windowSize);

#endif //__BINARIZE_UTILS_H__
