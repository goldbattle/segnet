/**
 * MIT License
 * Copyright (c) 2018 Patrick Geneva
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef AUGMENTATIONS_H
#define AUGMENTATIONS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>

// Random number generator
static std::uniform_real_distribution<double> unif(0, 1);
static std::uniform_real_distribution<double> unif_pn(-1, 1);
static std::mt19937_64 rng;

/**
 * The image should be rotated a bit to handle variations in the mount
 * Most camera should point forward in the car, but we should handle small variations
 */
inline void random_rotate(cv::Mat &cv_rgb, cv::Mat &cv_label) {

  // Store the original size
  cv::Size original_size = cv_rgb.size();
  cv::Point2f center(original_size.width / 2, original_size.height / 2);

  // rotate!
  // We generate a rotation, and then from that we want to zoom in to remove
  // Any of the black areas that the rotation has introduced....
  // This is a combination of many online sources as it seems nobody directly solved cleanly
  // TODO: this seems to segfault if we go to large of a rotation as the size goes to zero...
  // https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L41-L63
  // https://stackoverflow.com/a/16778797
  // https://stackoverflow.com/a/27137047
  // https://stackoverflow.com/a/56933590
  double angle = 10.0 * unif_pn(rng);
  cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
  double h = original_size.height;
  double w = original_size.width;
  double v_tan = std::abs(std::tan(angle * M_PI / 180.0));
  int b = int(v_tan / (1 - v_tan * v_tan) * (h - w * v_tan));
  int d = int(v_tan / (1 - v_tan * v_tan) * (w - h * v_tan));
  int nW = original_size.width - 2 * b;
  int nH = original_size.height - 2 * d;
  M.at<double>(0, 2) += (nW / 2) - center.x;
  M.at<double>(1, 2) += (nH / 2) - center.y;
  cv::warpAffine(cv_rgb, cv_rgb, M, cv::Size(nW, nH), cv::INTER_CUBIC);
  cv::warpAffine(cv_label, cv_label, M, cv::Size(nW, nH), cv::INTER_NEAREST);

  // Resize to the original
  cv::resize(cv_rgb, cv_rgb, original_size, 0, 0, cv::INTER_CUBIC);
  cv::resize(cv_label, cv_label, original_size, 0, 0, cv::INTER_NEAREST);
}

/**
 * This will randomly crop into a portion of the image
 * We shouldn't zoom to far into it as we should be semi-realistic
 */
inline void random_crop(cv::Mat &cv_rgb, cv::Mat &cv_label) {

  // Store the original size
  cv::Size original_size = cv_rgb.size();

  // crop image is at max 1/2 total width
  int crop_width = std::floor(cv_rgb.cols / 2.0 + cv_rgb.cols / 2.0 * unif(rng));
  int crop_height = std::floor(cv_rgb.rows / 2.0 + cv_rgb.rows / 2.0 * unif(rng));

  // randomly move it inside the total image
  int max_x = std::floor(cv_rgb.cols - crop_width);
  int max_y = std::floor(cv_rgb.rows - crop_height);
  int top_x = std::floor(max_x * unif(rng));
  int top_y = std::floor(max_y * unif(rng));

  // Actually perform the crop
  // int top_x = std::floor(cv_rgb.cols / 2.0 * unif(rng));
  // int top_y = std::floor(cv_rgb.rows / 2.0 * unif(rng));
  // int crop_width = (int)std::floor(cv_rgb.cols / 2.0 + cv_rgb.cols / 2.0 * unif(rng)) - top_x;
  // int crop_height = (int)std::floor(cv_rgb.rows / 2.0 + cv_rgb.rows / 2.0 * unif(rng)) - top_y;
  cv::Rect crop_area(top_x, top_y, crop_width, crop_height);
  cv_rgb = cv_rgb(crop_area);
  cv_label = cv_label(crop_area);

  // Apply random horizontal flips etc...
  if (unif(rng) > 0.5) {
    cv::flip(cv_rgb, cv_rgb, 1);
    cv::flip(cv_label, cv_label, 1);
  }

  // Resize to the original
  cv::resize(cv_rgb, cv_rgb, original_size, 0, 0, cv::INTER_CUBIC);
  cv::resize(cv_label, cv_label, original_size, 0, 0, cv::INTER_NEAREST);
}

/**
 * This will apply a random camera model change
 * We should change the focal point, camera center, and distortion.
 * TODO: We should sometimes use fisheye camera modeling...
 * TODO: Are the camera intrinsics reasonable?
 * https://github.com/albumentations-team/albumentations/blob/89a675cbfb2b76f6be90e7049cd5211cb08169a5/albumentations/augmentations/functional.py#L1048-L1085
 */
inline void random_camera_model(cv::Mat &cv_rgb, cv::Mat &cv_label) {

  // Store the original size
  cv::Size original_size = cv_rgb.size();

  // Camera intrinsic matrices
  cv::Mat cam(3, 3, cv::DataType<float>::type);
  cam.at<float>(0, 0) = original_size.width - 100 + 80 * unif_pn(rng);
  cam.at<float>(0, 1) = 0.0f;
  cam.at<float>(0, 2) = (original_size.width - 100 * unif_pn(rng)) / 2;
  cam.at<float>(1, 0) = 0.0f;
  cam.at<float>(1, 1) = original_size.height - 100 + 80 * unif_pn(rng);
  cam.at<float>(1, 2) = (original_size.height - 100 * unif_pn(rng)) / 2;
  cam.at<float>(2, 0) = 0.0f;
  cam.at<float>(2, 1) = 0.0f;
  cam.at<float>(2, 2) = 1.0f;
  cv::Mat dist(5, 1, cv::DataType<float>::type);
  dist.at<float>(0, 0) = 0.1 * unif_pn(rng);
  dist.at<float>(1, 0) = 0.05 * unif_pn(rng);
  dist.at<float>(2, 0) = 1e-3 * unif_pn(rng);
  dist.at<float>(3, 0) = 1e-4 * unif_pn(rng);
  dist.at<float>(4, 0) = 1e-5 * unif_pn(rng);

  // Distortion map
  cv::Mat map1, map2;
  cv::initUndistortRectifyMap(cam, dist, cv::Mat(), cam, original_size, CV_32FC1, map1, map2);
  cv::remap(cv_rgb, cv_rgb, map1, map2, cv::INTER_CUBIC, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::remap(cv_label, cv_label, map1, map2, cv::INTER_NEAREST, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

  // Resize to the original
  cv::resize(cv_rgb, cv_rgb, original_size, 0, 0, cv::INTER_CUBIC);
  cv::resize(cv_label, cv_label, original_size, 0, 0, cv::INTER_NEAREST);
}

/**
 * This will apply random value changes to the intensity.
 * Should only be applied to the RGB image!
 * Examples: blur, noise, shift rgb, jpeg compression, brightness,
 * Examples: gamma, gray scaling, sharpen, contrast
 */
inline void random_disturbances(cv::Mat &cv_rgb) {

  // Alpha beta change
  // https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
  double alpha = 0.3 * unif_pn(rng) + 1.0;
  double beta = 3 * unif(rng);
  cv_rgb.convertTo(cv_rgb, -1, alpha, beta);

  // Random gamma
  // https://docs.opencv.org/3.4/Basic_Linear_Transform_Tutorial_gamma.png
  double gamma = 0.05 * unif_pn(rng) + 1.0;
  cv::Mat lookUpTable(1, 256, CV_8U);
  uchar *p = lookUpTable.ptr();
  for (int i = 0; i < 256; ++i) {
    p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
  }
  cv::LUT(cv_rgb, lookUpTable, cv_rgb);

  // Blur the image
  if (unif(rng) > 0.5) {
    cv::GaussianBlur(cv_rgb, cv_rgb, cv::Size(5, 5), 0, 0);
  }

  // Add random grain perturbations
  cv::Mat grain(cv_rgb.rows, cv_rgb.cols, CV_8UC3);
  cv::randu(grain, cv::Scalar(-10, -10, -10), cv::Scalar(10, 10, 10));
  cv_rgb += grain;
}

#endif /* AUGMENTATIONS_H */
