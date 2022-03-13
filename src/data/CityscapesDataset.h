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

#ifndef DATA_CITYSCAPES_H
#define DATA_CITYSCAPES_H

#include <torch/data/datasets/base.h>
#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <dirent.h> // linux only
#include <functional>
#include <map>
#include <random>
#include <string>

#include "utils/augmentations.h"

/**
 * Dataset loader for cityscapes dataset
 * This will load data that is in the cityscapes format
 * https://www.cityscapes-dataset.com/
 */
class CityscapesDataset : public torch::data::Dataset<CityscapesDataset> {

public:
  // Mapping between our cityscape class IDs and the ones we will optimize
  std::map<char, char> map_class2id{
      {7, 1},  // 7: road
      {23, 2}, // 23: sky
      {26, 3}  // 26: car
  };

  // Mapping between our optimized IDs and our cityscape class
  std::map<char, char> map_id2class{
      {0, 0},  // 0: unlabeled
      {1, 7},  // 7: road
      {2, 23}, // 23: sky
      {3, 36}  // 26: car
  };

public:
  // The mode in which the dataset is loaded.
  enum class ModeDataSplit { kTrain, kTest };

  // The what type of groundtruth data is desired
  enum class ModeGroundtruth { kSegmentation, kDepth };

  // Random number generator
  std::mt19937_64 rng;

  // Our strings of image data
  std::vector<std::string> paths_rgb;

  // Our strings of segmentation labels
  std::vector<std::string> paths_labels_seg;

  // Our strings of disparity and camera params
  std::vector<std::string> paths_labels_disp;
  std::vector<std::string> paths_labels_cam;

  // If we should add randomized things to the images
  bool randomize;

  // What groundtruth mode we are using
  ModeGroundtruth groundtruth_mode;

  // Default constructor, should load directory
  CityscapesDataset(std::string pathroot, ModeDataSplit mode = ModeDataSplit::kTrain,
                    ModeGroundtruth modegt = ModeGroundtruth::kSegmentation, bool randomize = false);

  // Will get a single example for this dataset
  torch::data::Example<> get(size_t index) override {
    if (groundtruth_mode == ModeGroundtruth::kSegmentation)
      return get_segmentation_sample(index);
    else if (groundtruth_mode == ModeGroundtruth::kDepth)
      return get_depth_sample(index);
    else
      std::exit(EXIT_FAILURE);
  }

  // Return the total size of the dataset
  torch::optional<size_t> size() const override { return paths_rgb.size(); }

private:
  // Function that gets the image and labeled segmentations
  torch::data::Example<> get_segmentation_sample(size_t index);

  // Function that gets the image and depth in meters
  torch::data::Example<> get_depth_sample(size_t index);

  // Function that will get a list of all files in the specified directory
  // Taken from: https://stackoverflow.com/a/47045240
  void listFiles(const std::string &path, std::function<void(const std::string &)> cb) {
    if (auto dir = opendir(path.c_str())) {
      while (auto f = readdir(dir)) {
        if (!f->d_name || f->d_name[0] == '.')
          continue;
        if (f->d_type == DT_DIR)
          listFiles(path + f->d_name + "/", cb);

        if (f->d_type == DT_REG)
          cb(path + f->d_name);
      }
      closedir(dir);
    }
  }

  // Check if string ends with another string
  // Taken from: https://stackoverflow.com/a/2072890
  inline bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size())
      return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }
};

#endif // DATA_CITYSCAPES_H