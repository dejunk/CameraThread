#include <iostream>
#include <zconf.h>
#include <chrono>
#include <fstream>
#include <queue>

#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void initializeCamera();

void query_maximum_resolution(cv::VideoCapture *camera, int &max_width, int &max_height);

double frameDifference(cv::Mat &matFrameCurrent, cv::Mat &matFramePrevious);

void cameraRecord();

void cameraLoop();

void action();

boost::thread threadCamera, threadRecord;
pthread_mutex_t _mutexFrameCam1Last, _mutexFrameCam2Last;
cv::VideoCapture stream1;
uint64_t timestampcamera_ns;
cv::Mat matFrameForward;
std::ofstream lframe;
bool time_to_exit;
int max_width, max_height;
std::queue<uint64_t> qTime;
std::queue<cv::Mat> qFrame;

void initializeCamera() {
    stream1 = cv::VideoCapture(1);
//    query_maximum_resolution(&stream1, max_width, max_height);
//    max_width = 1280; max_height = 720;
//    max_width = 640; max_height = 480;
    max_width = 848; max_height = 480;

    //initialize Record folder
    boost::filesystem::path dir("./record_data");
    boost::filesystem::path dir2("./record_data/cam0");

    if (!(boost::filesystem::exists(dir))) {
        std::cout << "Doesn't Exists" << std::endl;
        if (boost::filesystem::create_directory(dir))
            std::cout << "....Successfully Created " << "./record_data/" << " Directory!" << std::endl;
    }
    if (!(boost::filesystem::exists(dir2))) {
        std::cout << "Doesn't Exists" << std::endl;
        if (boost::filesystem::create_directory(dir2))
            std::cout << "....Successfully Created " << "./record_data/cam0/" << " Directory!" << std::endl;
    }

    lframe.open("./record_data/frame.csv");
    lframe << "timestamp" << "\n";
}

//find maximum resolution
void query_maximum_resolution(cv::VideoCapture *camera, int &max_width, int &max_height) {
    // Save current resolution
    const int current_width = static_cast<int>(camera->get(CV_CAP_PROP_FRAME_WIDTH));
    const int current_height = static_cast<int>(camera->get(CV_CAP_PROP_FRAME_HEIGHT));

    // Get maximum resolution
    camera->set(CV_CAP_PROP_FRAME_WIDTH, 10000);
    camera->set(CV_CAP_PROP_FRAME_HEIGHT, 10000);
    max_width = static_cast<int>(camera->get(CV_CAP_PROP_FRAME_WIDTH));
    max_height = static_cast<int>(camera->get(CV_CAP_PROP_FRAME_HEIGHT));

    // Restore resolution
    camera->set(CV_CAP_PROP_FRAME_WIDTH, current_width);
    camera->set(CV_CAP_PROP_FRAME_HEIGHT, current_height);
}

// check is 2 frames is difference or not
double frameDifference(cv::Mat &matFrameCurrent, cv::Mat &matFramePrevious) {
    double diff = 0.0;
    assert(matFrameCurrent.rows > 0 && matFrameCurrent.cols > 0);
    assert(
            matFrameCurrent.rows == matFramePrevious.rows
            && matFrameCurrent.cols == matFramePrevious.cols);
    assert(
            matFrameCurrent.type() == CV_8U && matFramePrevious.type() == CV_8U);
    for (int i = 0; i < matFrameCurrent.rows; i++) {
        for (int j = 0; j < matFrameCurrent.cols; j++) {
            diff += matFrameCurrent.at<cv::Vec3b>(i, j)[1] - matFramePrevious.at<cv::Vec3b>(i, j)[1];
        }
    }
    return diff;
}

void cameraLoop() {
    int totalFrame = 0;

    stream1.set(CV_CAP_PROP_FRAME_WIDTH, max_width);
    stream1.set(CV_CAP_PROP_FRAME_HEIGHT, max_height);
//    stream1.set(CV_CAP_PROP_CONVERT_RGB , false);
    while (!time_to_exit) {
        timestampcamera_ns = boost::lexical_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        pthread_mutex_lock(&_mutexFrameCam1Last);

        stream1 >> matFrameForward;
        matFrameForward.convertTo(matFrameForward, CV_8U);
        cv::cvtColor(matFrameForward, matFrameForward, CV_BGR2GRAY);

        qFrame.push(matFrameForward);
        qTime.push(timestampcamera_ns);

        pthread_mutex_unlock(&_mutexFrameCam1Last);
//
        std::cout << "read matFrameForward size : " << matFrameForward.size() << std::endl;
        cv::imshow("Camera", matFrameForward);
        if (cv::waitKey(1) >= 0) break;

        totalFrame++;
        usleep(800000);
    }
    std::cout << "#Frame = " << totalFrame << std::endl;
}

void cameraRecord() {

    int totalRecord = 0;
    cv::Mat recFrameForward, lastestFrameForward;
    uint64_t timestampcamera;

    while (!time_to_exit || !qFrame.empty()) {
        //std::cout << "";
        if (matFrameForward.cols != max_width) continue;
        if (!qFrame.empty()) {

            int OldPrio = 0;
            pthread_mutex_setprioceiling(&_mutexFrameCam1Last, 0, &OldPrio);
            pthread_mutex_lock(&_mutexFrameCam1Last);

            recFrameForward = qFrame.front();
            timestampcamera = qTime.front();

            pthread_mutex_unlock(&_mutexFrameCam1Last);

            imwrite("./record_data/cam0/" + std::to_string(timestampcamera) + ".png", recFrameForward);
            lframe << timestampcamera << "\n";
            totalRecord++;

            recFrameForward.copyTo(lastestFrameForward);

            qFrame.pop();
            qTime.pop();
        }
    }
    std::cout << "#Record = " << totalRecord << std::endl;
}

void action() {
    sleep(20);
}

void istart() {
    // initilize camera parameter
    initializeCamera();

    //create camera thread
    std::cout << "Start camera thread..." << std::endl;
    threadCamera = boost::thread(&cameraLoop);

    // create record thread
    std::cout << "Start record thread..." << std::endl;
    threadRecord = boost::thread(&cameraRecord);
}

void istop() {
    //join thread
    time_to_exit = true;
    lframe.close();

    threadCamera.join();
    threadRecord.join();
    std::cout << "Finish recording." << std::endl;
}

int main() {
    std::cout << "Hello! This program is made for record the stream of images" << std::endl;
    time_to_exit = false;

    istart();

    //call action function
    action();

    istop();

    return 0;
}
