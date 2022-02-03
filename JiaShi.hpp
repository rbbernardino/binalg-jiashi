//
// Created by rodrigo on 23/01/2022.
//

#ifndef JIASHI_HPP
#define JIASHI_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "doxa/Parameters.hpp"
#include "doxa/Image.hpp"

using namespace cv;


namespace DibAlgs {
    class JiaShi {
    public:
        static Mat ToBinary(const cv::Mat &grayImageIn, const Doxa::Parameters &allParameters) {
            Mat binImageOut(grayImageIn.rows, grayImageIn.cols, CV_8UC1);

            // binarize with Doxa Niblack implementation
            Doxa::Parameters niblackParameters(Doxa::ParameterMap({
                                                                          {"window", allParameters.Get("niblack_window",
                                                                                                       60)},
                                                                          {"k",      allParameters.Get("niblack_k",
                                                                                                       0.2)}})
            );
            const Doxa::Image grayImageIn_doxa = Doxa::Image(grayImageIn.cols, grayImageIn.rows, grayImageIn.data);
            Doxa::Image binImageNiblack_doxa = Doxa::Niblack::ToBinaryImage(grayImageIn_doxa, niblackParameters);

            // convert Doxa image type to opencv type
            Mat binImageNiblack(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);
            std::memcpy(binImageNiblack.data, binImageNiblack_doxa.data, binImageNiblack_doxa.size);

            // dilate
            Mat dilatedNiblack;
            Mat structuralElement = getStructuringElement(MORPH_RECT, Size(3,3));
            dilate(binImageNiblack, dilatedNiblack, structuralElement);

            // estimate the background
            Mat estimatedBG = FindEstimatedBackground(grayImageIn, dilatedNiblack);

            imwrite("../test/4estimated_BG_5.png", estimatedBG);

//            binImageOut.data = binImageNiblack.data;
//            std::memcpy(binImageOut.data, binImageNiblack.data, binImageNiblack.size);
            return estimatedBG;
        }

    private:
        static Mat FindEstimatedBackground(Mat I_originalImage, const Mat IM_inpaintingMask) {
            long I_height = I_originalImage.rows;
            long I_width = I_originalImage.cols;
            Mat P_vet[4] = {
                    Mat(I_height, I_width, CV_8UC1),
                    Mat(I_height, I_width, CV_8UC1),
                    Mat(I_height, I_width, CV_8UC1),
                    Mat(I_height, I_width, CV_8UC1)
            };
            Mat M_tempImage(I_height, I_width, CV_8UC1);
            Mat M_tempImage_original;
            for (auto P_i: P_vet) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    for (auto y_i = 0; y_i < I_height; y_i++) {
                        if (IM_inpaintingMask.at<uchar>(y_i, x_i) == 0) {
                            M_tempImage.at<uchar>(y_i, x_i) = 0;
                        } else if (IM_inpaintingMask.at<uchar>(y_i, x_i) == 255) {
                            M_tempImage.at<uchar>(y_i, x_i) = 1;
                        } else {
                            cout << "ERROR: inpainting mask has pixel of value different from 255 and 0!" << endl;
                            exit(1);
                        }
                    }
                }
                M_tempImage_original = M_tempImage.clone();
                for (auto y_i = 0; y_i < I_height; y_i++) {
                    for (auto x_i = 0; x_i < I_width; x_i++) {
                        /**
                         * Mxy = 0  =>  Ixy = black (foreground)
                         * Mxy = 1  =>  Ixy = white (background)
                         */
                        if (M_tempImage.at<uchar>(y_i, x_i) == 0) {
                            uchar I_top = I_originalImage.at<uchar>(y_i - 1, x_i);
                            uchar M_top = M_tempImage.at<uchar>(y_i - 1, x_i);

                            uchar I_bottom = I_originalImage.at<uchar>(y_i + 1, x_i);
                            uchar M_bottom = M_tempImage.at<uchar>(y_i + 1, x_i);

                            uchar I_left = I_originalImage.at<uchar>(y_i, x_i - 1);
                            uchar M_left = M_tempImage.at<uchar>(y_i, x_i - 1);

                            uchar I_right = I_originalImage.at<uchar>(y_i, x_i + 1);
                            uchar M_right = M_tempImage.at<uchar>(y_i, x_i + 1);

                            // foreground pixel surrounded by foreground pixel
                            if(M_top + M_bottom + M_left + M_right == 0) {
                                P_i.at<uchar>(y_i, x_i) =  I_originalImage.at<uchar>(y_i, x_i);
                            } else {
                                P_i.at<uchar>(y_i, x_i) = (
                                                                  I_top * M_top +
                                                                  I_bottom * M_bottom +
                                                                  I_left * M_left +
                                                                  I_right * M_right
                                                          ) / (M_top + M_bottom + M_left + M_right);
                            }
                            I_originalImage.at<uchar>(y_i, x_i) = P_i.at<uchar>(y_i, x_i);
                            M_tempImage.at<uchar>(y_i, x_i) = 1;
                        } else {
//                            P_i.at<uchar>(y_i, x_i) = I_originalImage.at<uchar>(y_i, x_i);
                        }
                    }
                }
            }

            imwrite("../test/5A_P_1.png", P_vet[0]);
            imwrite("../test/5A_P_2.png", P_vet[1]);
            imwrite("../test/5A_P_3.png", P_vet[2]);
            imwrite("../test/5A_P_4.png", P_vet[3]);

            Mat estimatedBG(I_height, I_width, CV_8UC1);
            for (auto y_i = 0; y_i < I_height; y_i++) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    uchar min_Pixy = 255;
                    for(auto P_i : P_vet) {
                        auto cur_min = P_i.at<uchar>(y_i, x_i);
                        if(cur_min < min_Pixy)
                            min_Pixy = cur_min;
                    }
//                    if(M_tempImage_original.at<uchar>(y_i, x_i) == 0) {
                        estimatedBG.at<uchar>(y_i, x_i) = min_Pixy;
//                    } else {
//                        estimatedBG.at<uchar>(y_i, x_i) = I_originalImage.at<uchar>(y_i, x_i);
//                    }
                }
            }
            return estimatedBG;
        }
    };
}

#endif //JIASHI_HPP
