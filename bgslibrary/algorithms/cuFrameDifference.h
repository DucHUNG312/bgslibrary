#pragma once

#include "cuIBGS.h"

namespace bgslibrary {
namespace algorithms {
class cuFrameDifference : public cuIBGS {
private:
  int threshold;

public:
  cuFrameDifference();
  ~cuFrameDifference();

  void process(const cv::cuda::GpuMat &img_input, cv::cuda::GpuMat &img_output);

private:
  void save_config(cv::FileStorage &fs);
  void load_config(cv::FileStorage &fs);
};

cubgs_register(cuFrameDifference);
} // namespace algorithms
} // namespace bgslibrary
