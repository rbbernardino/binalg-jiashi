//
// Created by rodrigo on 23/01/2022.
//

#ifndef JIASHI_HPP
#define JIASHI_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "doxa/Parameters.hpp"
#include "doxa/Image.hpp"

// for debug
#include <iostream>
#include <fstream>

using namespace cv;


namespace DibAlgs {
    class JiaShi {
    public:
        static Mat ToBinary(const cv::Mat &grayImageIn, const Doxa::Parameters &allParameters) {
            Mat binImageOut(grayImageIn.rows, grayImageIn.cols, CV_8UC1);

            Mat gradientMap = ComputeGradientMap(grayImageIn, allParameters);

            float E_evalVet[256];
            multiApplyEvaluationFunction(gradientMap, E_evalVet);

            return gradientMap;
        }

    private:
        static void multiApplyEvaluationFunction(Mat gradientMap, float E_evalVet[]) {
            Mat binGradientVet[256];
            Mat f_discriminantVet[256];
            for (int t_threshold = 0; t_threshold < 256; t_threshold++) {
                binGradientVet[t_threshold] = Mat(gradientMap.rows, gradientMap.cols, CV_8UC1);
                threshold(gradientMap, binGradientVet[t_threshold], t_threshold, 1, CV_8UC1);

                f_discriminantVet[t_threshold] = Mat(gradientMap.rows, gradientMap.cols, CV_8UC1);

                int f_sum = 0;
                for (auto y_i = 0; y_i < gradientMap.rows; y_i++) {
                    for (auto x_i = 0; x_i < gradientMap.cols; x_i++) {
                        // check N_xy, neighborhood 8-connected pixels, looking for at least one non-edge pixel
                        uchar pixel_topleft = binGradientVet[t_threshold].at<uchar>(y_i - 1, x_i - 1);
                        uchar pixel_top = binGradientVet[t_threshold].at<uchar>(y_i - 1, x_i);
                        uchar pixel_topright = binGradientVet[t_threshold].at<uchar>(y_i - 1, x_i + 1);
                        uchar pixel_right = binGradientVet[t_threshold].at<uchar>(y_i, x_i + 1);
                        uchar pixel_bottomright = binGradientVet[t_threshold].at<uchar>(y_i + 1, x_i + 1);
                        uchar pixel_bottom = binGradientVet[t_threshold].at<uchar>(y_i + 1, x_i);
                        uchar pixel_bottomleft = binGradientVet[t_threshold].at<uchar>(y_i + 1, x_i - 1);
                        uchar pixel_left = binGradientVet[t_threshold].at<uchar>(y_i, x_i - 1);

                        // if at least one 0 is present => AND operation will make the result turns 0
                        bool has_nonedge_in_Nxy =
                                pixel_topleft && pixel_top && pixel_left && pixel_right && pixel_bottomleft &&
                                pixel_bottom && pixel_bottomright && pixel_topright;

                        // calculates f_edge => the discriminant function
                        uchar cur_gradientBinXy = binGradientVet[t_threshold].at<uchar>(y_i, x_i);
                        if (cur_gradientBinXy == 1 && has_nonedge_in_Nxy)
                            f_discriminantVet[t_threshold].at<uchar>(y_i, x_i) = 1;
                        else
                            f_discriminantVet[t_threshold].at<uchar>(y_i, x_i) = 0;
                        f_sum += f_discriminantVet[t_threshold].at<uchar>(y_i, x_i);
                    }
                }
                //                imwrite("../test/multi-grad/binGradient_t" + std::to_string(t_threshold) + ".png", binGradient_t);

                // calculates N_cc => the number of connected components in the image
                // by default, OpenCV uses 8-way connected components
                Mat labelImage(binGradientVet[t_threshold].size(), CV_32S);
                int N_connected_comp = cv::connectedComponents(binGradientVet[t_threshold], labelImage);

                E_evalVet[t_threshold] = (float) f_sum / (float) N_connected_comp;

                // ===================== DEBUG ==========================
//            cout << "t: " << t_threshold << "   f_sum: " << f_sum << "    Ncc: " << N_connected_comp << endl;
//                if (t_threshold == 65) {
//                    saveConnectedComponentsImage(N_connected_comp, binGradientVet[t_threshold], labelImage);
//                    Mat tempGradientBin(gradientMap.rows, gradientMap.cols, CV_8UC1);
//                    threshold(gradientMap, tempGradientBin, t_threshold, 255, CV_8UC1);
//                    imwrite("../test/ex1_gradientMap.png", gradientMap);
//                    imwrite("../test/ex1_gradientBin_t65.png", tempGradientBin);
//                }
            }
            // ===================== DEBUG ==========================
            // save E(t) values => histogram for plotting later
//            ofstream out_csv;
//            out_csv.open("../test/8_Et_histogram.csv");
//            out_csv << "t,Et" << endl;
//            for(auto t = 0; t < 256; t++) {
//                out_csv << t << "," << E_evalVet[t] << endl;
//            }
        }

        /**
         * For debugging purposes
         *     image generation from OpenCV docs:
         *     https://docs.opencv.org/4.x/de/d01/samples_2cpp_2connected_components_8cpp-example.html#a3
         */

        static void saveConnectedComponentsImage(int nLabels, Mat img, Mat labelImage) {
            cout << "Connected components: " << nLabels << endl;
            std::vector<Vec3b> colors(nLabels);
            colors[0] = Vec3b(0, 0, 0);//background
            for (int label = 1; label < nLabels; ++label) {
                colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
            }
            Mat dst(img.size(), CV_8UC3);
            for (int r = 0; r < dst.rows; ++r) {
                for (int c = 0; c < dst.cols; ++c) {
                    int label = labelImage.at<int>(r, c);
                    Vec3b &pixel = dst.at<Vec3b>(r, c);
                    pixel = colors[label];
                }
            }
            imwrite("../test/connected.png", dst);
        }

        static Mat ComputeGradientMap(const cv::Mat &grayImageIn, const Doxa::Parameters &allParameters) {
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

//            imwrite("../test/3a_niblack.png", binImageNiblack);
//            imwrite("../test/3b_inverted_niblack.png", invertedNiblack);
//            imwrite("../test/3c_dilated_inverted_niblack.png", dilatedNiblack);
//            imwrite("../test/3d_dilated_restored_niblack.png", dilatedNiblackRestored);
//            imwrite("../test/3e_estimated_BG.png", estimatedBG);

            Mat normalizedI(grayImageIn.rows, grayImageIn.cols,
//                             CV_32FC1);
                            CV_8UC1);
            NormalizeImage(grayImageIn, estimatedBG, normalizedI);

//            imwrite("../test/4_normalized_ex6.png", normalizedI);


            // o artigo diz usar o operador Schaar, porém eu testei com ele e os resultados não batem
            // Talvez seja a normalizacão "c" mencionada para converter o resultado do gradiente em 0~255
            // Ele detalha dizendo qual o valor de "c", mas usei o conversor padrão do OpenCV
            // Como o operador Sobel dá um resultado mais próximo do que é mostrado noa artigo, resolvi usar ele
            Mat gradientMap;
//            ApplyModifiedSobel(normalizedI, gradientMap, 0.5);
            ApplyOpenCVSobel(normalizedI, gradientMap);

//            imwrite("../test/5_gradientMap.png", gradientMap);

            return gradientMap;
        }

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

//            imwrite("../test/P_1.png", P1_lrtb);
//            imwrite("../test/P_2.png", P2_lrbt);
//            imwrite("../test/P_3.png", P3_rltb);
//            imwrite("../test/P_4.png", P4_rlbt);

            return estimatedBG;
        }

        static void NormalizeImage(Mat I_originalImage, Mat B_estimatedBackground, Mat &dst) {
            long I_height = I_originalImage.rows;
            long I_width = I_originalImage.cols;

            for (auto y_i = 0; y_i < I_height; y_i++) {
                for (auto x_i = 0; x_i < I_width; x_i++) {
                    uchar current_I = I_originalImage.at<uchar>(y_i, x_i);
                    uchar current_B = B_estimatedBackground.at<uchar>(y_i, x_i);
                    if (current_B > 0 && current_I < current_B) {
//                        dst.at<float>(y_i, x_i) = 255.0 * current_I / current_B;
                        dst.at<uchar>(y_i, x_i) = 255 * current_I / current_B;
                    } else {
//                        dst.at<float>(y_i, x_i) = 255;
                        dst.at<uchar>(y_i, x_i) = 255;
                    }
                }
            }
        }

        static void ApplyOpenCVSobel(const Mat &src, Mat &dst) {
            Mat gradientMapX, gradientMapY, absGradientMapX, absGradientMapY;
//            cv::Scharr(src, gradientMapX, CV_32F, 1, 0);
//            cv::Scharr(src, gradientMapY, CV_32F, 0, 1);

            cv::Sobel(src, gradientMapX, CV_32F, 1, 0);
            cv::Sobel(src, gradientMapY, CV_32F, 0, 1);

            cv::convertScaleAbs(gradientMapX, absGradientMapX);
            cv::convertScaleAbs(gradientMapY, absGradientMapY);

            cv::addWeighted(absGradientMapX, 0.5, absGradientMapY, 0.5, 0, dst);
        }

        static void ApplyModifiedSobel(const Mat &src, Mat &dst, const float c_normalizer) {
            Mat G_x = Mat::zeros(3, 3, CV_32F);
            G_x.at<float>(0, 0) = -3;
            G_x.at<float>(0, 2) = 3;
            G_x.at<float>(1, 0) = -10;
            G_x.at<float>(1, 2) = 10;
            G_x.at<float>(2, 0) = -3;
            G_x.at<float>(2, 2) = 3;

            Mat G_y = Mat::zeros(3, 3, CV_32F);
            G_y.at<float>(0, 0) = -3;
            G_y.at<float>(2, 0) = 3;
            G_y.at<float>(0, 1) = -10;
            G_y.at<float>(2, 1) = 10;
            G_y.at<float>(0, 2) = -3;
            G_y.at<float>(2, 2) = 3;

            // ddepth = -1 ==> same as source, which is this case will be CV_32F
            Mat Iconv_x(src.rows, src.cols, CV_32FC1);
            filter2D(src, Iconv_x, -1, G_x); //, Point(-1,-1), 0, BORDER_DEFAULT);

            Mat Iconv_y(src.rows, src.cols, CV_32FC1);
            filter2D(src, Iconv_y, -1, G_y); //, Point(-1,-1), 0, BORDER_DEFAULT);

            // a "L1-norm" é a soma dos valores absolutos das matrizes
            Mat abs_Iconv_y, abs_Iconv_x;
            cv::convertScaleAbs(Iconv_y, abs_Iconv_y);
            cv::convertScaleAbs(Iconv_x, abs_Iconv_x);
            cv::addWeighted(abs_Iconv_y, c_normalizer, abs_Iconv_x, c_normalizer, 0, dst);
//            dst = (abs_Iconv_x + abs_Iconv_y) * 0.5;

            // --------------------------------------------------------
//            double minVal;
//            double maxVal;
//            Point minLoc;
//            Point maxLoc;
//            minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);
//            cout << "min val: " << minVal << endl;
//            cout << "max val: " << maxVal << endl;
        }
    };
}

#endif //JIASHI_HPP
