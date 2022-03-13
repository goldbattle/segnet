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

#include "Comma10kDataset.h"

Comma10kDataset::Comma10kDataset(std::string pathroot, ModeDataSplit mode, bool randomize) {

  // Set if we should randomize
  this->randomize = randomize;

  // Open the file
  std::string path = pathroot + "files_trainable";
  std::ifstream file;
  file.open(path);
  if (!file) {
    std::cerr << "ERROR: Unable to open trainable file listing" << std::endl;
    std::cerr << path << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Loop through and generate our file listings
  // We are validating on images ending with "9.png"
  std::string line_mask;
  while (std::getline(file, line_mask)) {
    std::string line_rgb = std::regex_replace(line_mask, std::regex("masks"), "imgs");
    line_rgb = std::regex_replace(line_rgb, std::regex("masks2"), "imgs2");
    if (mode == Comma10kDataset::ModeDataSplit::kTest && ends_with(line_mask, "9.png")) {
      paths_rgb.push_back(pathroot + line_rgb);
      paths_labels_seg.push_back(pathroot + line_mask);
    } else if (mode == Comma10kDataset::ModeDataSplit::kTrain && !ends_with(line_mask, "9.png")) {
      paths_rgb.push_back(pathroot + line_rgb);
      paths_labels_seg.push_back(pathroot + line_mask);
    }
  }
  file.close();

  // Sort them so they match
  std::sort(paths_rgb.begin(), paths_rgb.end());
  std::sort(paths_labels_seg.begin(), paths_labels_seg.end());

  // Random order (ensure same random shuffle on both)
  // https://stackoverflow.com/a/16968342
  if (randomize) {
    unsigned int seed = std::time(NULL);
    std::srand(seed);
    std::random_shuffle(paths_rgb.begin(), paths_rgb.end());
    std::srand(seed);
    std::random_shuffle(paths_labels_seg.begin(), paths_labels_seg.end());
    std::srand(std::time(NULL));
  }

  // Check that they are of the same size
  if (paths_rgb.size() != paths_labels_seg.size()) {
    std::cerr << "number of RGB images does not equal CLASS label images" << std::endl;
    std::cerr << "rgb image count = " << paths_rgb.size() << std::endl;
    std::cerr << "label image count = " << paths_labels_seg.size() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Debug
  std::cout << "done loading dataset..." << std::endl;
  std::cout << "    - rgb image count = " << paths_rgb.size() << std::endl;
  std::cout << "    - label image count = " << paths_labels_seg.size() << std::endl;
}

torch::data::Example<> Comma10kDataset::get(size_t index) {

  // Assert that our index is in bound
  assert(index < paths_rgb.size());

  // Get the images
  std::string path_rgb = paths_rgb.at(index);
  std::string path_label = paths_labels_seg.at(index);

  // Load with the image from disk
  cv::Mat cv_rgb = cv::imread(path_rgb, cv::IMREAD_COLOR);
  cv::Mat cv_label_rgb = cv::imread(path_label, cv::IMREAD_COLOR);
  std::vector<cv::Mat> cv_label_channels;
  cv::split(cv_label_rgb, cv_label_channels);
  cv::Mat cv_label = cv_label_channels[0];

  // Assert that we have images
  assert(cv_rgb.rows != 0 && cv_rgb.cols != 0);
  assert(cv_label.rows != 0 && cv_label.cols != 0);
  assert(cv_rgb.rows == cv_label.rows && cv_rgb.cols == cv_label.cols);

  // Randomly apply transformations to our image
  if (randomize) {
    random_rotate(cv_rgb, cv_label);
    random_camera_model(cv_rgb, cv_label);
    random_crop(cv_rgb, cv_label);
    random_disturbances(cv_rgb);
  }

  // Resize the images
  cv::resize(cv_rgb, cv_rgb, cv::Size(640 / 2, 480 / 2), 0, 0, cv::INTER_CUBIC);
  cv::resize(cv_label, cv_label, cv::Size(640 / 2, 480 / 2), 0, 0, cv::INTER_NEAREST);

  // Our new label matrix that have all cityscape classes coverted
  // This will convert from the -1 to 32 classes to our 0-3 range
  cv::Mat cv_labelids = cv::Mat(cv_label.rows, cv_label.cols, CV_8UC1, cv::Scalar(0));
  for (int r = 0; r < cv_label.rows; r++) {
    for (int c = 0; c < cv_label.cols; c++) {
      if (map_class2id.find(cv_label.at<char>(r, c)) != map_class2id.end()) {
        cv_labelids.at<char>(r, c) = map_class2id.at(cv_label.at<char>(r, c));
      }
    }
  }

  // Debug image
  //  cv::imshow("test 1", cv_label);
  //  cv::imshow("test 2", (255 / map_id2class.size()) * cv_labelids);
  //  cv::waitKey(0);

  // Convert to pytorch tensor (needs to be [C,H,W] as per conv2d definition)
  auto input_ = torch::tensor(at::ArrayRef<uint8_t>(cv_rgb.data, cv_rgb.rows * cv_rgb.cols * 3)).view({cv_rgb.rows, cv_rgb.cols, 3});
  auto label_ = torch::tensor(at::ArrayRef<uint8_t>(cv_labelids.data, cv_labelids.rows * cv_labelids.cols * 1))
                    .view({cv_labelids.rows, cv_labelids.cols, 1});

  // Note that opencv stores things in [rgb,row,col] so we need to flip it after loading it into the tensor
  // Our view is of the "outer most" dimension, so we start with the number of columns, then rows, then the rgb
  input_ = input_.permute({2, 0, 1}).clone();
  label_ = label_.permute({2, 0, 1}).clone();

  // Convert our tensors to float and long types
  // Also we scale our input rbg image to be between [0..1]
  input_ = input_.to(torch::kFloat);
  input_ = input_ / 255.0;
  label_ = label_.to(torch::kLong);

  //  // Debug code to check that we are reading labels in correctly (should be in 0-5 range)
  //  auto foo_a = label_.accessor<long, 3>();
  //  for (int i = 0; i < foo_a.size(1); i++) {
  //    for (int j = 0; j < foo_a.size(2); j++) {
  //      if (foo_a[0][i][j] < 0 || foo_a[0][i][j] > 3) {
  //        std::cout << 0 << "," << i << "," << j << " = " << label_[0][i][j] << std::endl;
  //      }
  //    }
  //  }

  // Return our input and target
  return torch::data::Example<>(input_, label_);
}
