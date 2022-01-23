#include "utils.h"
#include "binalgorithms.h"

// OBS: capture execution time using <chrono> lib, default since C++11
bool binarizeImage(Mat imageIN, string fnameOUT, int algCode, int windowSize) {
    Mat imageOUT;
    Mat grayImageIN, grayImageTemp;
    cv::cvtColor(imageIN, grayImageIN, cv::COLOR_BGR2GRAY);
    const Doxa::Image image = Doxa::Image(grayImageIN.cols, grayImageIN.rows, grayImageIN.data);

    // // Setup base params for Doxa algorithms
    Doxa::Parameters parameters(Doxa::ParameterMap({{"window", DEFAULT_WINDOW_SIZE}}));

    // grayscale conversion repeated to measure time
    Doxa::Image binaryImage;
    long timeInNano;
    switch (algCode) {
        case AlgCode::Otsu: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            binaryImage = Doxa::Otsu::ToBinaryImage(image);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Bernsen: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            if (windowSize > 0) {
                parameters.Set("window", windowSize);
            } else {
                parameters.Set("window", BERNSEN_WINDOW);
            }
            parameters.Set("k", BERNSEN_K);
            binaryImage = Doxa::Bernsen::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Niblack: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            if (windowSize > 0) {
                parameters.Set("window", windowSize);
            } else {
                parameters.Set("window", NIBLACK_WINDOW);
            }
            parameters.Set("k", NIBLACK_K);
            binaryImage = Doxa::Niblack::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Sauvola: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            if (windowSize > 0) {
                parameters.Set("window", windowSize);
            } else {
                parameters.Set("window", SAUVOLA_WINDOW);
            }
            parameters.Set("k", SAUVOLA_K);
            binaryImage = Doxa::Sauvola::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Wolf: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            if (windowSize > 0) {
                parameters.Set("window", windowSize);
            } else {
                parameters.Set("window", WOLF_WINDOW);
            }
            parameters.Set("k", WOLF_K);
            binaryImage = Doxa::Wolf::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Nick: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            if (windowSize > 0) {
                parameters.Set("window", windowSize);
            } else {
                parameters.Set("window", NICK_WINDOW);
            }
            parameters.Set("k", NICK_K);
            binaryImage = Doxa::Nick::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::GatosBeta: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            parameters.Set("k", GATOS_K);
            parameters.Set("glyph", GATOS_GLYPH);
            binaryImage = Doxa::Gatos::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::SuLu: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            binaryImage = Doxa::Su::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Singh: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            parameters.Set("k", SINGH_K);
            binaryImage = Doxa::TRSingh::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::Bataineh: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            binaryImage = Doxa::Bataineh::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::WAN: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            parameters.Set("k", WAN_K);
            binaryImage = Doxa::Wan::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
        case AlgCode::ISauvola: {
            auto start = std::chrono::high_resolution_clock::now();
            cv::cvtColor(imageIN, grayImageTemp, cv::COLOR_BGR2GRAY);
            parameters.Set("k", ISAUVOLA_K);
            binaryImage = Doxa::ISauvola::ToBinaryImage(image, parameters);
            auto finish = std::chrono::high_resolution_clock::now();
            imageOUT = Mat(image.height, image.width, CV_8UC1);
            imageOUT.data = binaryImage.data;
            timeInNano = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            break;
        }
    }
    std::cout << timeInNano << ";";
    std::cout << "-2" << ";";
    imwrite(fnameOUT, imageOUT);
    return true;
}
