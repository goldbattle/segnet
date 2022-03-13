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

#ifndef DATA_COMMAN10K_H
#define DATA_COMMAN10K_H

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
#include <regex>
#include <string>

#include "utils/augmentations.h"

/**
 * Dataset loader for comma10k dataset
 * https://github.com/commaai/comma10k
 */
class Comma10kDataset : public torch::data::Dataset<Comma10kDataset> {

public:
  /**
   * Mapping between comma ids and the ones we optimize
   * 1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
   * 2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
   * 3 - #808060 - undrivable
   * 4 - #00ff66 - movable (vehicles and people/animals)
   * 5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)
   */
  std::map<char, char> map_class2id = {
      {32, 0},  // 32: road (x20)
      {00, 1},  // 0: lane markings (x00)
      {96, 2},  // 96: undrivable (x60)
      {102, 3}, // 102: movable (x66)
      {255, 4}, // 26: my car (xff)
  };

  /**
   * Mapping between optimize ids and comma ids (used for visualization)
   * We use the last channel as a grayscale
   * 1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
   * 2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
   * 3 - #808060 - undrivable
   * 4 - #00ff66 - movable (vehicles and people/animals)
   * 5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)
   */
  std::map<char, char> map_id2class = {
      //{0, 0},  // 0: unlabeled
      {0, 32},  // 32: road (x20)
      {1, 00},  // 0: lane markings (x00)
      {2, 96},  // 96: undrivable (x60)
      {3, 102}, // 102: movable (x66)
      {4, 255}, // 26: my car (xff)
  };

  /**
   * Mapping between optimize ids and hex RGB values (used for visualization)
   * NOTE: in order of BLUE, GREEN, RED to match opencv
   * 1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
   * 2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
   * 3 - #808060 - undrivable
   * 4 - #00ff66 - movable (vehicles and people/animals)
   * 5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)
   */
  std::map<char, cv::Vec3b> map_id2hex = {
      {0, cv::Vec3b(32, 32, 64)},   // 32: road (x20)
      {1, cv::Vec3b(0, 0, 255)},    // 0: lane markings (x00)
      {2, cv::Vec3b(96, 128, 128)}, // 96: undrivable (x60)
      {3, cv::Vec3b(102, 255, 0)},  // 102: movable (x66)
      {4, cv::Vec3b(255, 0, 204)},  // 26: my car (xff)
  };

public:
  // The mode in which the dataset is loaded.
  enum class ModeDataSplit { kTrain, kTest };

  // Our strings of image data
  std::vector<std::string> paths_rgb;

  // Our strings of segmentation labels
  std::vector<std::string> paths_labels_seg;

  // If we should add randomized things to the images
  bool randomize;

  // Default constructor, should load directory
  Comma10kDataset(std::string pathroot, ModeDataSplit mode = ModeDataSplit::kTrain, bool randomize = false);

  // Will get a single example for this dataset
  torch::data::Example<> get(size_t index) override;

  // Return the total size of the dataset
  torch::optional<size_t> size() const override { return paths_rgb.size(); }

private:
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

#endif // DATA_COMMAN10K_H