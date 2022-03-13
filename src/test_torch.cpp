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
#include <opencv2/opencv.hpp>

#include "network/models/UNetModel.h"

int main() {

  // Create the network(RGB image = > class masks)
  size_t n_channels = 3;
  size_t n_classes = 4;
  UNetModel net(n_channels, n_classes);
  std::cout << net << std::endl;

  // Debug print the network
  for (const auto &pair : net->named_parameters()) {
    std::cout << pair.key() << std::endl;
  }

  // Test a random vector
  torch::Tensor input = torch::ones({1, 3, 480, 640});
  torch::Tensor output = net->forward(input);
  std::cout << "input - " << input.sizes() << std::endl;
  std::cout << "output - " << output.sizes() << std::endl << std::endl;

  char x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  cv::Mat X(2, 2, CV_8UC3, x);
  std::cout << "testing opencv to torch conversions..." << std::endl;
  std::cout << X << std::endl;
  std::cout << X.size << std::endl;

  std::cout << "opencv at() access" << std::endl;
  std::cout << "0,0 = " << X.at<cv::Vec3b>(0, 0) << std::endl;
  std::cout << "0,1 = " << X.at<cv::Vec3b>(0, 1) << std::endl;
  std::cout << "1,0 = " << X.at<cv::Vec3b>(1, 0) << std::endl;
  std::cout << "1,1 = " << X.at<cv::Vec3b>(1, 1) << std::endl;

  std::cout << "opencv data access" << std::endl;
  std::cout << "0,0 = " << (int)X.data[0] << ", " << (int)X.data[1] << ", " << (int)X.data[2] << std::endl;
  std::cout << "0,1 = " << (int)X.data[3 + 0] << ", " << (int)X.data[3 + 1] << ", " << (int)X.data[3 + 2] << std::endl;
  std::cout << "1,0 = " << (int)X.data[6 + 0] << ", " << (int)X.data[6 + 1] << ", " << (int)X.data[6 + 2] << std::endl;
  std::cout << "1,1 = " << (int)X.data[9 + 0] << ", " << (int)X.data[9 + 1] << ", " << (int)X.data[9 + 2] << std::endl;

  auto input_ = torch::tensor(at::ArrayRef<uint8_t>(X.data, X.rows * X.cols * 3)).view({X.rows, X.cols, 3}).to(torch::kFloat);
  std::cout << "torch read in" << std::endl;
  std::cout << input_ << std::endl;

  std::cout << "torch perturbed (red, blue, and green channels)" << std::endl;
  input_ = input_.permute({2, 0, 1}).clone();
  std::cout << input_ << std::endl;

  std::cout << "opencv from torch data" << std::endl;
  input_ = input_.permute({1, 2, 0}).clone();
  input_ = input_.to(torch::kInt8);
  cv::Mat Xnew2(2, 2, CV_8UC3, input_.data_ptr<int8_t>());
  std::cout << Xnew2 << std::endl;
  std::cout << Xnew2.size << std::endl;

  std::cout << "0,0 = " << (int)Xnew2.data[0] << ", " << (int)Xnew2.data[1] << ", " << (int)Xnew2.data[2] << std::endl;
  std::cout << "0,1 = " << (int)Xnew2.data[3 + 0] << ", " << (int)Xnew2.data[3 + 1] << ", " << (int)Xnew2.data[3 + 2] << std::endl;
  std::cout << "1,0 = " << (int)Xnew2.data[6 + 0] << ", " << (int)Xnew2.data[6 + 1] << ", " << (int)Xnew2.data[6 + 2] << std::endl;
  std::cout << "1,1 = " << (int)Xnew2.data[9 + 0] << ", " << (int)Xnew2.data[9 + 1] << ", " << (int)Xnew2.data[9 + 2] << std::endl;
}
