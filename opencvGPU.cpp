#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static string joinPath(string a, const string& b) {
    if (!a.empty() && a.back() != '/' && a.back() != '\\') a += "/";
    return a + b;
}

static string makeTimestampName(const string& ext) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    tm = *std::localtime(&t);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf), "record_%04d%02d%02d_%02d%02d%02d.%s",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec, ext.c_str());
    return string(buf);
}

int main(int argc, char** argv) {
    string data_dir = (argc > 1) ? string(argv[1]) : string("data");
    string classes_path = joinPath(data_dir, "object_detection_classes_coco.txt");
    string pb_path = joinPath(data_dir, "frozen_inference_graph.pb");
    string pbtxt_path = joinPath(data_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt");

    cout << "[INFO] Using data_dir: " << data_dir << "\n";

    vector<string> class_names;
    {
        ifstream ifs(classes_path);
        if (!ifs.is_open()) {
            cerr << "[ERROR] Failed to open classes file: " << classes_path << "\n";
            return 1;
        }
        string line;
        while (getline(ifs, line)) if (!line.empty()) class_names.push_back(line);
    }

    Net net;
    try {
        net = readNetFromTensorflow(pb_path, pbtxt_path);
    }
    catch (const cv::Exception& e) {
        cerr << "[ERROR] readNetFromTensorflow failed:\n" << e.what() << "\n";
        return 1;
    }

    // Prefer GPU if available
    try {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        cout << "[INFO] DNN backend: CUDA\n";
    }
    catch (const cv::Exception&) {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        cout << "[INFO] DNN backend: CPU\n";
    }

    // ---- Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Could not open webcam.\n";
        return 1;
    }

    // (Optional) force resolution for ¡°full¡± wide view
    // cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    // cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    // ---- Setup recorder
    Size frameSize((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    double camFps = cap.get(CAP_PROP_FPS);
    if (camFps < 1.0 || camFps > 240.0) camFps = 30.0; // many webcams report 0; pick a sane default

    VideoWriter writer;
    bool recording = true;

    auto openWriter = [&]() {
        // Try MP4 first (smaller), then fallback to AVI MJPG (most reliable)
        string mp4Name = makeTimestampName("mp4");
        int fourcc_mp4 = VideoWriter::fourcc('m', 'p', '4', 'v');
        if (writer.open(mp4Name, fourcc_mp4, camFps, frameSize, true)) {
            cout << "[INFO] Recording to " << mp4Name << " (mp4v)\n";
            return;
        }

        string aviName = makeTimestampName("avi");
        int fourcc_avi = VideoWriter::fourcc('M', 'J', 'P', 'G');
        if (writer.open(aviName, fourcc_avi, camFps, frameSize, true)) {
            cout << "[INFO] Recording to " << aviName << " (MJPG)\n";
            return;
        }

        cerr << "[ERROR] Could not open VideoWriter (mp4 and avi both failed).\n";
        };

    openWriter(); // start recording immediately

    const float min_confidence_score = 0.5f;

    while (cap.isOpened()) {
        Mat image;
        if (!cap.read(image) || image.empty()) break;

        int h = image.rows;
        int w = image.cols;

        int64 start = getTickCount();

        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(), true, false);
        net.setInput(blob);
        Mat output = net.forward();

        int64 end = getTickCount();
        double sec = (end - start) / getTickFrequency();
        double fps = (sec > 1e-9) ? (1.0 / sec) : 0.0;

        Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        for (int i = 0; i < results.rows; i++) {
            int class_id = (int)results.at<float>(i, 1);
            float confidence = results.at<float>(i, 2);
            if (confidence < min_confidence_score) continue;

            int left = (int)(results.at<float>(i, 3) * w);
            int top = (int)(results.at<float>(i, 4) * h);
            int right = (int)(results.at<float>(i, 5) * w);
            int bottom = (int)(results.at<float>(i, 6) * h);

            left = max(0, min(left, w - 1));
            right = max(0, min(right, w - 1));
            top = max(0, min(top, h - 1));
            bottom = max(0, min(bottom, h - 1));

            rectangle(image, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

            string label = "class_id=" + to_string(class_id);
            if (class_id >= 1 && class_id <= (int)class_names.size())
                label = class_names[class_id - 1];

            label += " " + to_string((int)(confidence * 100)) + "%";
            putText(image, label, Point(left, max(20, top - 10)),
                FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
        }

        // HUD
        putText(image, "FPS: " + to_string((int)fps), Point(30, 40),
            FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 255, 0), 2);

        if (recording) {
            putText(image, "REC", Point(30, 80),
                FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        // ---- Write frame (record annotated video)
        if (recording && writer.isOpened()) {
            // Ensure the frame matches writer size exactly
            if (image.size() == frameSize) writer.write(image);
            else {
                Mat resized;
                resize(image, resized, frameSize);
                writer.write(resized);
            }
        }

        imshow("opencvGPU", image);

        int k = waitKey(1);
        if (k == 'q' || k == 27) break;            // quit
        if (k == 'r') {                             // toggle recording
            recording = !recording;
            cout << "[INFO] recording = " << (recording ? "ON" : "OFF") << "\n";
            if (recording && !writer.isOpened()) openWriter();
            if (!recording && writer.isOpened()) writer.release();
        }
    }

    if (writer.isOpened()) writer.release();
    cap.release();
    destroyAllWindows();
    return 0;
}
