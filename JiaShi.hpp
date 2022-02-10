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
                                                                                                       -0.2)}})
            );
            const Doxa::Image grayImageIn_doxa = Doxa::Image(grayImageIn.cols, grayImageIn.rows, grayImageIn.data);
            Doxa::Image binImageNiblack_doxa = Doxa::Niblack::ToBinaryImage(grayImageIn_doxa, niblackParameters);

            // convert Doxa image type to opencv type
            Mat binImageNiblack(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);
            std::memcpy(binImageNiblack.data, binImageNiblack_doxa.data, binImageNiblack_doxa.size);

            // invert before dilate
            Mat invertedNiblack(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);
            bitwise_not(binImageNiblack, invertedNiblack);

            // dilate
            Mat dilatedNiblack;
            Mat structuralElement = getStructuringElement(MORPH_RECT, Size(3, 3));
            dilate(invertedNiblack, dilatedNiblack, structuralElement);

            // invert back from dilate before BG estimation
            Mat dilatedNiblackRestored(grayImageIn_doxa.height, grayImageIn_doxa.width, CV_8UC1);
            bitwise_not(dilatedNiblack, dilatedNiblackRestored);

            // estimate the background
            Mat estimatedBG = FindEstimatedBackground(grayImageIn, dilatedNiblackRestored);

            imwrite("../test/3a_niblack.png", binImageNiblack);
            imwrite("../test/3b_inverted_niblack.png", invertedNiblack);
            imwrite("../test/3c_dilated_inverted_niblack.png", dilatedNiblack);
            imwrite("../test/3d_dilated_restored_niblack.png", dilatedNiblackRestored);
            imwrite("../test/3e_estimated_BG.png", estimatedBG);

//            binImageOut.data = binImageNiblack.data;
//            std::memcpy(binImageOut.data, binImageNiblack.data, binImageNiblack.size);
            return estimatedBG;
        }

    private:
        static Mat FindEstimatedBackground(Mat I_originalImage, const Mat IM_inpaintingMask) {
            long I_height = I_originalImage.rows;
            long I_width = I_originalImage.cols;

            Mat P1_lrtb = Mat(I_height, I_width, CV_8UC1);
            Mat P2_lrbt = Mat(I_height, I_width, CV_8UC1);
            Mat P3_rltb = Mat(I_height, I_width, CV_8UC1);
            Mat P4_rlbt = Mat(I_height, I_width, CV_8UC1);

            Mat M_tempImage_original(I_height, I_width, CV_8UC1);

            // convert grayscale binary (0, 255) to unary binary (0, 1)
            cv::threshold(IM_inpaintingMask, M_tempImage_original, 128, 1, cv::THRESH_BINARY);
//            for (auto y_i = 0; y_i < I_height; y_i++) {
//                for (auto x_i = 0; x_i < I_width; x_i++) {
//                    if (IM_inpaintingMask.at<uchar>(y_i, x_i) == 0) {
//                        M_tempImage.at<uchar>(y_i, x_i) = 0;
//                    } else if (IM_inpaintingMask.at<uchar>(y_i, x_i) == 255) {
//                        M_tempImage.at<uchar>(y_i, x_i) = 1;
//                    } else {
//                        cout << "ERROR: inpainting mask has pixel of value different from 255 and 0!" << endl;
//                        exit(1);
//                    }
//                }
//            }

            Mat M_tempImage = M_tempImage_original.clone();
            Mat I_1 = I_originalImage.clone();
            P1_lrtb = I_originalImage.clone();
            M_tempImage_original = M_tempImage.clone();
            for (auto y_i = 0; y_i < I_height; y_i++) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    /**
                     * Mxy = 0  =>  Ixy = black, 0 (foreground)
                     * Mxy = 1  =>  Ixy = white, 255 (background)
                     */
                    uchar M_current = M_tempImage.at<uchar>(y_i, x_i);
                    if (M_current == 0) {
                        uchar I_top = I_1.at<uchar>(y_i - 1, x_i);
                        uchar M_top = M_tempImage.at<uchar>(y_i - 1, x_i);

                        uchar I_bottom = I_1.at<uchar>(y_i + 1, x_i);
                        uchar M_bottom = M_tempImage.at<uchar>(y_i + 1, x_i);

                        uchar I_left = I_1.at<uchar>(y_i, x_i - 1);
                        uchar M_left = M_tempImage.at<uchar>(y_i, x_i - 1);

                        uchar I_right = I_1.at<uchar>(y_i, x_i + 1);
                        uchar M_right = M_tempImage.at<uchar>(y_i, x_i + 1);

                        // foreground pixel surrounded by foreground pixel
                        if (M_top + M_bottom + M_left + M_right == 0) {
                            P1_lrtb.at<uchar>(y_i, x_i) = I_1.at<uchar>(y_i, x_i);
                        } else {
                            P1_lrtb.at<uchar>(y_i, x_i) = (
                                                                  I_top * M_top +
                                                                  I_bottom * M_bottom +
                                                                  I_left * M_left +
                                                                  I_right * M_right
                                                          ) / (M_top + M_bottom + M_left + M_right);
                        }
                        I_1.at<uchar>(y_i, x_i) = P1_lrtb.at<uchar>(y_i, x_i);
                        M_tempImage.at<uchar>(y_i, x_i) = 1;
                    }
                }
            }

            M_tempImage = M_tempImage_original.clone();
            Mat I_2 = I_originalImage.clone();
            P2_lrbt = I_1.clone();
            M_tempImage_original = M_tempImage.clone();
            for (auto y_i = I_height - 1; y_i >= 0; y_i--) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    /**
                     * Mxy = 0  =>  Ixy = black, 0 (foreground)
                     * Mxy = 1  =>  Ixy = white, 255 (background)
                     */
                    uchar M_current = M_tempImage.at<uchar>(y_i, x_i);
                    if (M_current == 0) {
                        uchar I_top = I_2.at<uchar>(y_i - 1, x_i);
                        uchar M_top = M_tempImage.at<uchar>(y_i - 1, x_i);

                        uchar I_bottom = I_2.at<uchar>(y_i + 1, x_i);
                        uchar M_bottom = M_tempImage.at<uchar>(y_i + 1, x_i);

                        uchar I_left = I_2.at<uchar>(y_i, x_i - 1);
                        uchar M_left = M_tempImage.at<uchar>(y_i, x_i - 1);

                        uchar I_right = I_2.at<uchar>(y_i, x_i + 1);
                        uchar M_right = M_tempImage.at<uchar>(y_i, x_i + 1);

                        // foreground pixel surrounded by foreground pixel
                        if (M_top + M_bottom + M_left + M_right == 0) {
                            P2_lrbt.at<uchar>(y_i, x_i) = I_2.at<uchar>(y_i, x_i);
                        } else {
                            P2_lrbt.at<uchar>(y_i, x_i) = (
                                                                  I_top * M_top +
                                                                  I_bottom * M_bottom +
                                                                  I_left * M_left +
                                                                  I_right * M_right
                                                          ) / (M_top + M_bottom + M_left + M_right);
                        }
                        I_2.at<uchar>(y_i, x_i) = P2_lrbt.at<uchar>(y_i, x_i);
                        M_tempImage.at<uchar>(y_i, x_i) = 1;
                    }
                }
            }

            M_tempImage = M_tempImage_original.clone();
            Mat I_3 = I_originalImage.clone();
            P3_rltb = I_2.clone();
            M_tempImage_original = M_tempImage.clone();
            for (auto y_i = 0; y_i < I_height; y_i++) {
                for (auto x_i = I_width - 1; x_i > 0; x_i--) {
                    /**
                     * Mxy = 0  =>  Ixy = black, 0 (foreground)
                     * Mxy = 1  =>  Ixy = white, 255 (background)
                     */
                    uchar M_current = M_tempImage.at<uchar>(y_i, x_i);
                    if (M_current == 0) {
                        uchar I_top = I_3.at<uchar>(y_i - 1, x_i);
                        uchar M_top = M_tempImage.at<uchar>(y_i - 1, x_i);

                        uchar I_bottom = I_3.at<uchar>(y_i + 1, x_i);
                        uchar M_bottom = M_tempImage.at<uchar>(y_i + 1, x_i);

                        uchar I_left = I_3.at<uchar>(y_i, x_i - 1);
                        uchar M_left = M_tempImage.at<uchar>(y_i, x_i - 1);

                        uchar I_right = I_3.at<uchar>(y_i, x_i + 1);
                        uchar M_right = M_tempImage.at<uchar>(y_i, x_i + 1);

                        // foreground pixel surrounded by foreground pixel
                        if (M_top + M_bottom + M_left + M_right == 0) {
                            P3_rltb.at<uchar>(y_i, x_i) = I_3.at<uchar>(y_i, x_i);
                        } else {
                            P3_rltb.at<uchar>(y_i, x_i) = (
                                                                  I_top * M_top +
                                                                  I_bottom * M_bottom +
                                                                  I_left * M_left +
                                                                  I_right * M_right
                                                          ) / (M_top + M_bottom + M_left + M_right);
                        }
                        I_3.at<uchar>(y_i, x_i) = P3_rltb.at<uchar>(y_i, x_i);
                        M_tempImage.at<uchar>(y_i, x_i) = 1;
                    }
                }
            }

            M_tempImage = M_tempImage_original.clone();
            Mat I_4 = I_originalImage.clone();
            P4_rlbt = I_3.clone();
            M_tempImage_original = M_tempImage.clone();
            for (auto y_i = I_height - 1; y_i < I_height; y_i++) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    /**
                     * Mxy = 0  =>  Ixy = black, 0 (foreground)
                     * Mxy = 1  =>  Ixy = white, 255 (background)
                     */
                    uchar M_current = M_tempImage.at<uchar>(y_i, x_i);
                    if (M_current == 0) {
                        uchar I_top = I_4.at<uchar>(y_i - 1, x_i);
                        uchar M_top = M_tempImage.at<uchar>(y_i - 1, x_i);

                        uchar I_bottom = I_4.at<uchar>(y_i + 1, x_i);
                        uchar M_bottom = M_tempImage.at<uchar>(y_i + 1, x_i);

                        uchar I_left = I_4.at<uchar>(y_i, x_i - 1);
                        uchar M_left = M_tempImage.at<uchar>(y_i, x_i - 1);

                        uchar I_right = I_4.at<uchar>(y_i, x_i + 1);
                        uchar M_right = M_tempImage.at<uchar>(y_i, x_i + 1);

                        // foreground pixel surrounded by foreground pixel
                        if (M_top + M_bottom + M_left + M_right == 0) {
                            P4_rlbt.at<uchar>(y_i, x_i) = I_4.at<uchar>(y_i, x_i);
                        } else {
                            P4_rlbt.at<uchar>(y_i, x_i) = (
                                                                  I_top * M_top +
                                                                  I_bottom * M_bottom +
                                                                  I_left * M_left +
                                                                  I_right * M_right
                                                          ) / (M_top + M_bottom + M_left + M_right);
                        }
                        I_4.at<uchar>(y_i, x_i) = P4_rlbt.at<uchar>(y_i, x_i);
                        M_tempImage.at<uchar>(y_i, x_i) = 1;
                    }
                }
            }

            Mat estimatedBG(I_height, I_width, CV_8UC1);
            for (auto y_i = 0; y_i < I_height; y_i++) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    uchar min_Pxy = 255;

                    uchar pixel_P1xy = P1_lrtb.at<uchar>(y_i, x_i);
                    uchar pixel_P2xy = P2_lrbt.at<uchar>(y_i, x_i);
                    uchar pixel_P3xy = P3_rltb.at<uchar>(y_i, x_i);
                    uchar pixel_P4xy = P4_rlbt.at<uchar>(y_i, x_i);

                    if (pixel_P1xy < min_Pxy) min_Pxy = pixel_P1xy;
                    if (pixel_P2xy < min_Pxy) min_Pxy = pixel_P2xy;
                    if (pixel_P3xy < min_Pxy) min_Pxy = pixel_P3xy;
                    if (pixel_P4xy < min_Pxy) min_Pxy = pixel_P4xy;

                    if (M_tempImage_original.at<uchar>(y_i, x_i) == 0) {
                        estimatedBG.at<uchar>(y_i, x_i) = min_Pxy;
                    } else {
                        estimatedBG.at<uchar>(y_i, x_i) = I_originalImage.at<uchar>(y_i, x_i);
                    }
                }
            }

            imwrite("../test/P_1.png", P1_lrtb);
            imwrite("../test/P_2.png", P2_lrbt);
            imwrite("../test/P_3.png", P3_rltb);
            imwrite("../test/P_4.png", P4_rlbt);

            return estimatedBG;
        }
    };
}

#endif //JIASHI_HPP
