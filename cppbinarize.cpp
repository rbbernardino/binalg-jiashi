#include "utils.h"
#include "args.hxx"

int main(int argc, char *argv[]) {
    Mat imageIN;
    string fpathIN, fpathOUT;
    int algCode;
    int windowSize;

    string noteStr =
            "Supported Algorithms:\n"
            "Code  -  Name\n"
            "0 - Otsu\n"
            "1 - Bernsen\n"
            "2 - Niblack\n"
            "3 - Sauvola\n"
            "4 - Wolf\n"
            "5 - Nick\n"
            "6 - Gatos-Beta\n"
            "7 - Su-Lu\n"
            "8 - Singh\n"
            "9 - Bataineh\n"
            "10 - WAN\n"
            "11 - ISauvola\n"
            "12 - Jia-Shi\n"
            "Most algorithms implemented by Brandon M. Petty";
    args::ArgumentParser parser("This program is responsible for binarizing images with several algorithms. "
                                "The input and output may be in any format supported by OpenCV.",
                                noteStr);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> pFpathIN(parser, "imgIN", "The input image", {'i'});
    args::ValueFlag<std::string> pFpathOUT(parser, "outputName", "The output filename", {'o'});
    args::ValueFlag<int> pWindowSize(parser, "windowSize", "The window size for local methods", {'w'});
    args::ValueFlag<int> pAlgCode(parser, "algCode", "The algorithm to be used for binarization", {'a'});

    //----------------------------------------------------------------------------
    // ARGS PARSING
    try {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help) {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    if (!pFpathIN || !pFpathOUT || !pAlgCode) {
        std::cerr << "ERROR! Please specify the input output file and algorithm name." << std::endl;
        std::cerr << parser;
        return 1;
    }

    //----------------------------------------------------------------------------
    // IMAGE READING
    fpathIN = args::get(pFpathIN);
    fpathOUT = args::get(pFpathOUT);
    algCode = args::get(pAlgCode);
    imageIN = imread(fpathIN, IMREAD_COLOR);
    if (imageIN.data == NULL) {
        std::cout << "ERROR: couldn't read input image!" << std::endl;
        return 1;
    }

    try {
        if (pWindowSize) {
            windowSize = args::get(pWindowSize);
        } else {
            windowSize = -1;
        }
        if (!binarizeImage(imageIN, fpathOUT, algCode, windowSize)) {
            return 1;
        }
        imageIN.release();
    }
    catch (Exception e) {
        std::cerr << e.what() << std::endl;
        std::cerr << e.file << ": " << e.line << endl;
        return 1;
    }

    return 0;
}
