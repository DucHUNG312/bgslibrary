#include "cuFrameDifference.h"

using namespace bgslibrary::algorithms;

cuFrameDifference::cuFrameDifference()
    : cuIBGS(quote(cuFrameDifference)), enableThreshold(true), threshold(15) {
  debug_construction(cuFrameDifference);
  initLoadSaveConfig(algorithm_name);
}

cuFrameDifference::~cuFrameDifference() {
  debug_destruction(cuFrameDifference);
}

void cuFrameDifference::process(const cv::cuda::GpuMat &img_input,
                                cv::cuda::GpuMat &img_output,
                                cv::cuda::GpuMat &img_bgmodel) {
  init(img_input, img_output, img_bgmodel);

  if (img_background.empty()) {
    img_input.copyTo(img_background);
    img_background.copyTo(img_bgmodel);
    first_time = false;
    return;
  }

  cv::cuda::absdiff(img_background, img_input, img_foreground);

  if (img_foreground.channels() == 3) {
    cv::cuda::cvtColor(img_foreground, img_foreground, cv::COLOR_BGR2GRAY);
  }

  if (enableThreshold)
    cv::cuda::threshold(img_foreground, img_foreground, threshold, 255,
                        cv::THRESH_BINARY);

  img_foreground.copyTo(img_output);

  img_input.copyTo(img_background);
  img_background.copyTo(img_bgmodel);

  first_time = false;
}

void cuFrameDifference::save_config(cv::FileStorage &fs) {
  fs << "enableThreshold" << enableThreshold;
  fs << "threshold" << threshold;
}

void cuFrameDifference::load_config(cv::FileStorage &fs) {
  fs["enableThreshold"] >> enableThreshold;
  fs["threshold"] >> threshold;
}
