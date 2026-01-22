#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

static const vector<string> COCO80 = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

struct LetterBoxInfo {
    float scale;
    int pad_x;
    int pad_y;
    int new_w;
    int new_h;
};

static Mat letterbox(const Mat& src, int dst_w, int dst_h, LetterBoxInfo& info) {
    int w = src.cols, h = src.rows;
    float r = std::min((float)dst_w / (float)w, (float)dst_h / (float)h);
    int new_w = (int)std::round(w * r);
    int new_h = (int)std::round(h * r);

    Mat resized;
    resize(src, resized, Size(new_w, new_h));

    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;

    Mat out(dst_h, dst_w, src.type(), Scalar(114, 114, 114));
    resized.copyTo(out(Rect(pad_x, pad_y, new_w, new_h)));

    info.scale = r;
    info.pad_x = pad_x;
    info.pad_y = pad_y;
    info.new_w = new_w;
    info.new_h = new_h;
    return out;
}

static void drawLabel(Mat& img, const string& text, int left, int top) {
    int baseline = 0;
    Size ts = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    int y = max(top, ts.height + 6);
    rectangle(img, Point(left, y - ts.height - 8), Point(left + ts.width + 8, y + baseline + 2), Scalar(0, 0, 0), FILLED);
    putText(img, text, Point(left + 4, y - 4), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
}

static float IoU(const Rect2f& a, const Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return (uni <= 0.f) ? 0.f : (inter / uni);
}

struct Detection {
    Rect2f box;
    float score;
    int class_id;
};

struct Track {
    int id;
    Rect2f box;
    int class_id;
    float score;
    int age;      // frames since created
    int missed;   // consecutive frames without match
};

class SimpleIoUTracker {
public:
    SimpleIoUTracker(float iou_th = 0.3f, int max_missed = 30, float smooth = 0.7f)
        : iou_th_(iou_th), max_missed_(max_missed), smooth_(smooth) {
    }

    vector<Track> update(const vector<Detection>& dets) {
        // mark all tracks as missed by default
        for (auto& t : tracks_) {
            t.age++;
            t.missed++;
        }

        // build candidate pairs (track, det) with IoU
        struct Pair { int ti, di; float iou; };
        vector<Pair> pairs;
        pairs.reserve(tracks_.size() * dets.size());

        for (int ti = 0; ti < (int)tracks_.size(); ++ti) {
            for (int di = 0; di < (int)dets.size(); ++di) {
                // class-consistent tracking tends to be more stable
                if (tracks_[ti].class_id != dets[di].class_id) continue;
                float iou = IoU(tracks_[ti].box, dets[di].box);
                if (iou > 0.f) pairs.push_back({ ti, di, iou });
            }
        }

        // greedy match by IoU desc (simple + fast)
        sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.iou > b.iou; });

        vector<char> det_used(dets.size(), 0);
        vector<char> trk_used(tracks_.size(), 0);

        for (const auto& p : pairs) {
            if (p.iou < iou_th_) break;
            if (trk_used[p.ti] || det_used[p.di]) continue;

            // match
            auto& t = tracks_[p.ti];
            const auto& d = dets[p.di];

            // smooth box to reduce jitter
            t.box.x = smooth_ * t.box.x + (1.f - smooth_) * d.box.x;
            t.box.y = smooth_ * t.box.y + (1.f - smooth_) * d.box.y;
            t.box.width = smooth_ * t.box.width + (1.f - smooth_) * d.box.width;
            t.box.height = smooth_ * t.box.height + (1.f - smooth_) * d.box.height;

            t.score = d.score;
            t.class_id = d.class_id;
            t.missed = 0;

            trk_used[p.ti] = 1;
            det_used[p.di] = 1;
        }

        // new tracks for unmatched detections
        for (int di = 0; di < (int)dets.size(); ++di) {
            if (det_used[di]) continue;
            Track t;
            t.id = next_id_++;
            t.box = dets[di].box;
            t.class_id = dets[di].class_id;
            t.score = dets[di].score;
            t.age = 1;
            t.missed = 0;
            tracks_.push_back(t);
        }

        // remove dead tracks
        tracks_.erase(remove_if(tracks_.begin(), tracks_.end(),
            [&](const Track& t) { return t.missed > max_missed_; }), tracks_.end());

        // return active-ish tracks (recently seen)
        vector<Track> active;
        active.reserve(tracks_.size());
        for (auto& t : tracks_) {
            if (t.missed <= 2) active.push_back(t);
        }
        return active;
    }

    void reset() {
        tracks_.clear();
        next_id_ = 1;
    }

private:
    vector<Track> tracks_;
    int next_id_ = 1;
    float iou_th_;
    int max_missed_;
    float smooth_;
};

int main() {
    string modelPath = "data/yolov8s.onnx";

    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
    }
    catch (const cv::Exception& e) {
        cerr << "[ERROR] readNetFromONNX failed:\n" << e.what() << "\n";
        return 1;
    }

    // CUDA preferred, fallback to CPU
    try {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);        // try DNN_TARGET_CUDA_FP16 if you want
        cout << "[INFO] Using CUDA backend\n";
    }
    catch (...) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cout << "[INFO] Using CPU backend\n";
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Cannot open camera.\n";
        return 1;
    }

    // ---- Recording (reliable)
    Size frameSize((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    double outFps = cap.get(CAP_PROP_FPS);
    if (outFps < 1.0 || outFps > 240.0) outFps = 30.0;
    VideoWriter writer;
    bool recording = false;
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G'); // AVI MJPG: reliable on Windows

    // ---- Tracker
    SimpleIoUTracker tracker(/*iou_th=*/0.30f, /*max_missed=*/30, /*smooth=*/0.70f);

    const int inpSize = 640;

    // ¡°Cleaner boxes¡± defaults (good starting point)
    const float confTh = 0.25f;
    const float nmsTh = 0.45f;

    while (true) {
        Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        int64 t0 = getTickCount();

        LetterBoxInfo lb;
        Mat lbimg = letterbox(frame, inpSize, inpSize, lb);

        Mat blob = cv::dnn::blobFromImage(lbimg, 1.0 / 255.0, Size(inpSize, inpSize), Scalar(), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        if (outputs.empty()) break;

        Mat out = outputs[0];

        if (out.dims != 3) {
            cerr << "[ERROR] Unexpected output dims: " << out.dims << "\n";
            break;
        }

        // Handle [1,84,8400] or [1,8400,84] or [1,85,8400]/[1,8400,85]
        int d1 = out.size[1];
        int d2 = out.size[2];

        Mat det; // [N x C]
        if (d1 <= 90) { // channels-first case
            Mat m(d1, d2, CV_32F, out.ptr<float>());
            transpose(m, det); // [d2 x d1]
        }
        else {
            det = Mat(d1, d2, CV_32F, out.ptr<float>()).clone();
        }

        const int C = det.cols; // 84 or 85 typically
        if (C < 84) {
            cerr << "[ERROR] Unexpected det.cols: " << C << "\n";
            break;
        }

        const bool hasObj = (C == 85);
        const int clsStart = hasObj ? 5 : 4;   // [cx,cy,w,h,(obj), classes...]
        const int clsEnd = C;                  // exclusive

        vector<Detection> dets;
        dets.reserve(det.rows);

        for (int i = 0; i < det.rows; i++) {
            float cx = det.at<float>(i, 0);
            float cy = det.at<float>(i, 1);
            float w = det.at<float>(i, 2);
            float h = det.at<float>(i, 3);

            float obj = hasObj ? det.at<float>(i, 4) : 1.0f;

            // Find best class
            Point maxLoc;
            double maxScore;
            Mat cls = det.row(i).colRange(clsStart, clsEnd);
            minMaxLoc(cls, 0, &maxScore, 0, &maxLoc);

            int class_id = maxLoc.x;
            float score = obj * (float)maxScore;
            if (score < confTh) continue;

            float x1 = cx - 0.5f * w;
            float y1 = cy - 0.5f * h;

            // undo letterbox
            float ox1 = (x1 - lb.pad_x) / lb.scale;
            float oy1 = (y1 - lb.pad_y) / lb.scale;
            float ow = w / lb.scale;
            float oh = h / lb.scale;

            Rect2f r(ox1, oy1, ow, oh);

            // clamp to frame
            r.x = std::max(0.f, std::min(r.x, (float)frame.cols - 1.f));
            r.y = std::max(0.f, std::min(r.y, (float)frame.rows - 1.f));
            r.width = std::max(0.f, std::min(r.width, (float)frame.cols - r.x));
            r.height = std::max(0.f, std::min(r.height, (float)frame.rows - r.y));

            dets.push_back({ r, score, class_id });
        }

        // ---- Cleaner boxes: class-wise NMS
        vector<Detection> finalDet;
        finalDet.reserve(dets.size());

        // group by class
        vector<vector<int>> perClass(COCO80.size());
        for (int i = 0; i < (int)dets.size(); ++i) {
            int cid = dets[i].class_id;
            if (cid >= 0 && cid < (int)COCO80.size()) perClass[cid].push_back(i);
        }

        for (int cid = 0; cid < (int)perClass.size(); ++cid) {
            const auto& idxs = perClass[cid];
            if (idxs.empty()) continue;

            vector<Rect> boxes;
            vector<float> scores;
            boxes.reserve(idxs.size());
            scores.reserve(idxs.size());

            for (int id : idxs) {
                Rect2f rf = dets[id].box;
                boxes.emplace_back((int)std::round(rf.x), (int)std::round(rf.y),
                    (int)std::round(rf.width), (int)std::round(rf.height));
                scores.push_back(dets[id].score);
            }

            vector<int> keep;
            cv::dnn::NMSBoxes(boxes, scores, confTh, nmsTh, keep);

            for (int k : keep) {
                int detIndex = idxs[k];
                finalDet.push_back(dets[detIndex]);
            }
        }

        // ---- Tracking
        vector<Track> tracks = tracker.update(finalDet);

        // ---- Draw tracks
        for (const auto& t : tracks) {
            Rect r((int)std::round(t.box.x), (int)std::round(t.box.y),
                (int)std::round(t.box.width), (int)std::round(t.box.height));

            r &= Rect(0, 0, frame.cols, frame.rows);
            rectangle(frame, r, Scalar(0, 0, 255), 2);

            string cname = (t.class_id >= 0 && t.class_id < (int)COCO80.size()) ? COCO80[t.class_id] : "cls";
            string text = "#" + to_string(t.id) + " " + cname + " " + to_string((int)(t.score * 100)) + "%";
            drawLabel(frame, text, r.x, r.y);
        }

        // FPS
        double dt = (getTickCount() - t0) / getTickFrequency();
        double fps = (dt > 1e-9) ? (1.0 / dt) : 0.0;
        putText(frame, "FPS: " + to_string((int)fps), Point(30, 40),
            FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 255, 0), 2);

        // Recording overlay + write
        if (recording) {
            putText(frame, "REC", Point(30, 80), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 0, 255), 2);
            if (!writer.isOpened()) {
                writer.open("yolo_record.avi", fourcc, outFps, frameSize, true);
                cout << "[INFO] recording started -> yolo_record.avi\n";
            }
            if (writer.isOpened()) {
                if (frame.size() == frameSize) writer.write(frame);
                else {
                    Mat resized;
                    resize(frame, resized, frameSize);
                    writer.write(resized);
                }
            }
        }

        imshow("YOLOv8 OpenCV DNN CUDA + Tracking", frame);

        int k = waitKey(1);
        if (k == 27 || k == 'q') break;

        // toggle recording
        if (k == 'r') {
            recording = !recording;
            cout << "[INFO] recording = " << (recording ? "ON" : "OFF") << "\n";
            if (!recording && writer.isOpened()) {
                writer.release(); // finalizes AVI cleanly
                cout << "[INFO] recording stopped (file finalized)\n";
            }
        }

        // reset tracks
        if (k == 'c') {
            tracker.reset();
            cout << "[INFO] tracker reset\n";
        }
    }

    if (writer.isOpened()) writer.release();
    cap.release();
    destroyAllWindows();
    return 0;
}
