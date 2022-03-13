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

#include <iostream>
#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/augmentations.h"

int main() {

  // open the image
  cv::Mat image = cv::imread("/home/patrick/github/segnet/docs/image.png", cv::IMREAD_COLOR);
  cv::Mat mask = cv::imread("/home/patrick/github/segnet/docs/mask.png", cv::IMREAD_COLOR);
  cv::resize(image, image, cv::Size(640 / 2, 480 / 2), 0, 0, cv::INTER_CUBIC);
  cv::resize(mask, mask, cv::Size(640 / 2, 480 / 2), 0, 0, cv::INTER_NEAREST);
  cv::Mat imgout1;
  cv::hconcat(image, mask, imgout1);
  cv::imshow("original", imgout1);

  while (true) {

    //=====================================================================
    //=====================================================================

    // rotate the image
    cv::Mat rot_image = image.clone();
    cv::Mat rot_mask = mask.clone();
    random_rotate(rot_image, rot_mask);
    cv::Mat imgout2;
    cv::hconcat(rot_image, rot_mask, imgout2);
    cv::vconcat(imgout1, imgout2, imgout2);
    cv::imshow("rotated", imgout2);

    //=====================================================================
    //=====================================================================

    cv::Mat crop_image = image.clone();
    cv::Mat crop_mask = mask.clone();
    random_crop(crop_image, crop_mask);
    cv::Mat imgout3;
    cv::hconcat(crop_image, crop_mask, imgout3);
    cv::vconcat(imgout1, imgout3, imgout3);
    cv::imshow("cropped", imgout3);

    //=====================================================================
    //=====================================================================

    cv::Mat dist_image = image.clone();
    random_disturbances(dist_image);
    cv::Mat imgout4;
    cv::hconcat(dist_image, image, imgout4);
    cv::imshow("distorted", imgout4);

    //=====================================================================
    //=====================================================================

    cv::Mat cam_image = image.clone();
    cv::Mat cam_mask = mask.clone();
    random_camera_model(cam_image, cam_mask);
    cv::Mat imgout5;
    cv::hconcat(cam_image, cam_mask, imgout5);
    cv::vconcat(imgout1, imgout5, imgout5);
    cv::imshow("camera distortion", imgout5);

    //=====================================================================
    //=====================================================================

    // wait for the user to inspect
    cv::waitKey(200);
  }
}
