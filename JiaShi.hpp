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
        static Mat ToBinary(const Doxa::Image &grayImageIn_doxa, const Doxa::Parameters &allParameters) {
            Mat binImageOut(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);

            // binarize with Doxa Niblack implementation
            Doxa::Parameters niblackParameters(Doxa::ParameterMap({
                                                                          {"window", allParameters.Get("niblack_window",
                                                                                                       60)},
                                                                          {"k",      allParameters.Get("niblack_k",
                                                                                                       0.2)}})
            );
            Doxa::Image binImageNiblack_doxa = Doxa::Niblack::ToBinaryImage(grayImageIn_doxa, niblackParameters);

            // convert Doxa image type to opencv type
            Mat binImageNiblack(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);
            std::memcpy(binImageNiblack.data, binImageNiblack_doxa.data, binImageNiblack_doxa.size);

            // convert GrayImageIn image type to opencv type
            Mat grayImageIn(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);
            std::memcpy(grayImageIn.data, grayImageIn_doxa.data, grayImageIn_doxa.size);

            // dilate
            Mat dilatedNiblack;
            Mat structuralElement = getStructuringElement(MORPH_RECT, Size(3,3));
            dilate(binImageNiblack, dilatedNiblack, structuralElement);

            // estimate the background
            Mat estimatedBG = FindEstimatedBackground(grayImageIn, dilatedNiblack);

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
            for (auto P_i: P_vet) {
                for (auto i = 0; i < I_width; i++) {
                    for (auto j = 0; j < I_height; j++) {
                        if (IM_inpaintingMask.at<uchar>(j, i) == 0) {
                            M_tempImage.at<uchar>(j, i) = 0;
                        } else {
                            M_tempImage.at<uchar>(j, i) = 1;
                        }
                    }
                }
                for (auto y_i = 0; y_i < I_height; y_i++) {
                    for (auto x_i = 0; x_i < I_width; x_i++) {
                        if (M_tempImage.at<uchar>(y_i, x_i) == 0) {
                            uchar I_top = I_originalImage.at<uchar>(y_i - 1, x_i);
                            uchar M_top = M_tempImage.at<uchar>(y_i - 1, x_i);
                            uchar I_bottom = I_originalImage.at<uchar>(y_i + 1, x_i);
                            uchar M_bottom = M_tempImage.at<uchar>(y_i + 1, x_i);
                            uchar I_left = I_originalImage.at<uchar>(y_i, x_i - 1);
                            uchar M_left = M_tempImage.at<uchar>(y_i, x_i - 1);
                            uchar I_right = I_originalImage.at<uchar>(y_i + 1, x_i + 1);
                            uchar M_right = M_tempImage.at<uchar>(y_i + 1, x_i + 1);
                            P_i.at<uchar>(y_i, x_i) = (
                                                              I_top * M_top +
                                                              I_bottom * M_bottom +
                                                              I_left * M_left +
                                                              I_right * M_right
                                                      ) / 4;
                            I_originalImage.at<uchar>(y_i, x_i) = P_i.at<uchar>(y_i, x_i);
                            M_tempImage.at<uchar>(y_i, x_i) = 1;
                        }
                    }
                }
            }

            Mat estimatedBG(I_height, I_width, CV_8UC1);
            for (auto y_i = 0; y_i < I_height; y_i++) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    uchar min_Pixy = 255;
                    for(auto P_i : P_vet) {
                        auto cur_min = P_i.at<uchar>(y_i, x_i);
                        if(cur_min < min_Pixy)
                            min_Pixy = cur_min;
                    }
                    estimatedBG.at<uchar>(y_i, x_i) = min_Pixy;
                }
            }
            return estimatedBG;
        }
    };
}

#endif //JIASHI_HPP
