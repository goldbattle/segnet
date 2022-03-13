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

#include "CityscapesDataset.h"

CityscapesDataset::CityscapesDataset(std::string pathroot, ModeDataSplit mode, ModeGroundtruth modegt, bool randomize) {

  // What folder we are going into
  std::string modestr = (mode == ModeDataSplit::kTrain) ? "train" : "val";

  // Set if we should randomize
  this->randomize = randomize;
  this->groundtruth_mode = modegt;

  // rgb images
  listFiles(pathroot + "leftImg8bit/" + modestr + "/", [this](const std::string &path) { paths_rgb.push_back(path); });

  // label images (label id images only)
  listFiles(pathroot + "gtFine/" + modestr + "/", [this](const std::string &path) {
    if (ends_with(path, "labelIds.png")) {
      paths_labels_seg.push_back(path);
    }
  });

  // disparity images
  listFiles(pathroot + "disparity/" + modestr + "/", [this](const std::string &path) { paths_labels_disp.push_back(path); });

  // disparity images
  listFiles(pathroot + "camera/" + modestr + "/", [this](const std::string &path) { paths_labels_cam.push_back(path); });

  // Sort them so they match
  std::sort(paths_rgb.begin(), paths_rgb.end());
  std::sort(paths_labels_seg.begin(), paths_labels_seg.end());
  std::sort(paths_labels_disp.begin(), paths_labels_disp.end());
  std::sort(paths_labels_cam.begin(), paths_labels_cam.end());

  // Check that they are of the same size
  if (modegt == ModeGroundtruth::kSegmentation && paths_rgb.size() != paths_labels_seg.size()) {
    std::cerr << "number of RGB images does not equal CLASS label images" << std::endl;
    std::cerr << "rgb image count = " << paths_rgb.size() << std::endl;
    std::cerr << "label image count = " << paths_labels_seg.size() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check that they are of the same size
  if (modegt == ModeGroundtruth::kDepth && (paths_rgb.size() != paths_labels_disp.size() || paths_rgb.size() != paths_labels_cam.size())) {
    std::cerr << "number of RGB images does not equal DEPTH images" << std::endl;
    std::cerr << "rgb image count = " << paths_rgb.size() << std::endl;
    std::cerr << "depth disparity image count = " << paths_labels_disp.size() << std::endl;
    std::cerr << "cam param image count = " << paths_labels_cam.size() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Debug
  std::cout << "done loading dataset..." << std::endl;
  std::cout << "    - rgb image count = " << paths_rgb.size() << std::endl;
  std::cout << "    - label image count = " << paths_labels_seg.size() << std::endl;
  std::cout << "    - depth disparity count = " << paths_labels_disp.size() << std::endl;
  std::cout << "    - cam param count = " << paths_labels_cam.size() << std::endl;
}

torch::data::Example<> CityscapesDataset::get_segmentation_sample(size_t index) {

  // Assert that our index is in bound
  assert(index < paths_rgb.size());

  // Get the images
  std::string path_rgb = paths_rgb.at(index);
  std::string path_label = paths_labels_seg.at(index);

  // Load with the image from disk
  cv::Mat cv_rgb = cv::imread(path_rgb, cv::IMREAD_COLOR);
  cv::Mat cv_label = cv::imread(path_label, cv::IMREAD_GRAYSCALE);

  // Assert that we have images
  assert(cv_rgb.rows != 0 && cv_rgb.cols != 0);
  assert(cv_label.rows != 0 && cv_label.cols != 0);
  assert(cv_rgb.rows == cv_label.rows && cv_rgb.cols == cv_label.cols);

  // Randomly apply transformations to our image
  if (randomize) {
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
  // cv::imshow("test 1",cv_label);
  // cv::imshow("test 2",(255/map_id2class.size())*cv_labelids);
  // cv::waitKey(0);

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

  // Debug code to check that we are reading labels in correctly (should be in 0-3 range)
  // auto foo_a = label_.accessor<long,3>();
  // for(int i = 0; i < foo_a.size(1); i++) {
  //    for (int j = 0; j < foo_a.size(2); j++) {
  //        if(foo_a[0][i][j] < 0 || foo_a[0][i][j] > 3) {
  //            std::cout << 0 << "," << i << "," << j << " = " << label_[0][i][j] << std::endl;
  //        }
  //    }
  //}

  // Return our input and target
  return torch::data::Example<>(input_, label_);
}

torch::data::Example<> CityscapesDataset::get_depth_sample(size_t index) {

  std::cout << "depth not implemented completely..." << std::endl;
  std::exit(EXIT_FAILURE);

  //    // Assert that our index is in bound
  //    assert(index < paths_rgb.size());
  //
  //    // Get the images
  //    std::string path_rgb = paths_rgb.at(index);
  //    std::string path_disp = paths_labels_disp.at(index);
  //    std::string path_cam = paths_labels_cam.at(index);
  //
  //    // Load camera calibration json
  //    std::ifstream if_cam(path_cam.c_str());
  //    nlohmann::json cam_json;
  //    if_cam >> cam_json;
  //
  //    // Load with the image from disk
  //    cv::Mat cv_rgb = cv::imread(path_rgb, cv::IMREAD_COLOR);
  //    cv::Mat cv_disp = cv::imread(path_disp, cv::IMREAD_ANYDEPTH);
  //    cv_disp.convertTo(cv_disp, CV_32FC1);
  //
  //    // Assert that we have have images
  //    assert(cv_rgb.rows != 0 && cv_rgb.cols != 0);
  //    assert(cv_disp.rows != 0 && cv_disp.cols != 0);
  //    assert(cv_rgb.rows == cv_disp.rows && cv_rgb.cols == cv_disp.cols);
  //
  //    // Get the baseline and focal lengths from our calibration
  //    // depth = baseline * focal_length / disparity_value
  //    float baseline = cam_json["extrinsic"]["baseline"].get<float>();
  //    float fx = cam_json["intrinsic"]["fx"].get<float>();
  //    //std::cout << "baseline- " << baseline << " | fx- " << fx << std::endl;
  //
  //    // Convert our disparity maps into real metric depths.
  //    // We need the camera matrix to do this as it is a function fo the stereo matching
  //    cv::Mat cv_depth_meters = cv::Mat(cv_disp.rows,cv_disp.cols,CV_32FC1,cv::Scalar(0.0f));
  //    for(int r=0; r<cv_disp.rows; r++) {
  //        for (int c=0; c<cv_disp.cols; c++) {
  //            // To obtain the disparity values, compute for each pixel p with p > 0: d = ( float(p) - 1. ) / 256.,
  //            // While a value p = 0 is an invalid measurement.
  //            // Warning: the images are stored as 16-bit pngs, which is non-standard and not supported by all libraries.
  //            if(cv_disp.at<float>(r,c) > 0) {
  //                float val_diparity = (cv_disp.at<float>(r,c)-1)/256.0f;
  //                cv_depth_meters.at<float>(r,c) = (baseline*fx)/val_diparity;
  //                if(isinf(cv_depth_meters.at<float>(r,c))) cv_depth_meters.at<float>(r,c) = 0.0f;
  //                if(cv_depth_meters.at<float>(r,c) < 0) cv_depth_meters.at<float>(r,c) = 0.0f;
  //                if(cv_depth_meters.at<float>(r,c) > 200.0f) cv_depth_meters.at<float>(r,c) = 0.0f;
  //            }
  //        }
  //    }
  //
  //    // Debug image
  //    //double minVal1, maxVal1;
  //    //cv::minMaxLoc(cv_depth_meters, &minVal1, &maxVal1);
  //    //cv::Mat cv_depth_meters_temp = 255 * (cv_depth_meters - minVal1)/(maxVal1 - minVal1);
  //    //cv_depth_meters_temp.convertTo(cv_depth_meters_temp, CV_8UC1);
  //    //cv::imshow("test 1",cv_depth_meters_temp);
  //    //cv::waitKey(0);
  //
  //    // Randomly crop the image in a random aspect ratio
  //    std::uniform_real_distribution<double> unif(0,1);
  //    if(randomize) {
  //        // crop image is at max 1/3 total width
  //        int crop_width = std::floor(cv_rgb.cols/2.0+cv_rgb.cols/2.0*unif(rng));
  //        int crop_height = std::floor(cv_rgb.rows/2.0+cv_rgb.rows/2.0*unif(rng));
  //        // randomly move it inside the total image
  //        int max_x = std::floor(cv_rgb.cols - crop_width);
  //        int max_y = std::floor(cv_rgb.rows - crop_height);
  //        int top_x = std::floor(max_x*unif(rng));
  //        int top_y = std::floor(max_y*unif(rng));
  //        // Actually perform the crop
  //        cv::Rect crop_area(top_x,top_y,crop_width,crop_height);
  //        cv_rgb = cv_rgb(crop_area);
  //        cv_depth_meters = cv_depth_meters(crop_area);
  //    }
  //
  //    // Resize the images
  //    cv::resize(cv_rgb, cv_rgb, cv::Size(640/2,480/2), 0, 0, cv::INTER_CUBIC);
  //    cv::resize(cv_depth_meters, cv_depth_meters, cv::Size(640/2,480/2), 0, 0, cv::INTER_LINEAR);
  //
  //    // Apply random horizontal flips etc...
  //    if(randomize && unif(rng) > 0.5) {
  //        cv::flip(cv_rgb, cv_rgb, 1);
  //        cv::flip(cv_depth_meters, cv_depth_meters, 1);
  //    }
  //
  //    // Add random color to the image
  //    if(randomize) {
  //        cv::Mat imgran(cv_rgb.rows,cv_rgb.cols,CV_8UC3);
  //        cv::randu(imgran,cv::Scalar(-10,-10,-10),cv::Scalar(10,10,10));
  //        cv_rgb += imgran;
  //    }
  //
  //    // Enforce that our depth is in a reasonable range of values
  //    // This can be caused by our interpolation and other perturbations
  //    cv_depth_meters.setTo(0, cv_depth_meters > 200.0f);
  //    cv_depth_meters.setTo(0, cv_depth_meters < 0.0f);
  //
  //    // Convert to pytorch tensor (needs to be [C,H,W] as per conv2d definition)
  //    auto input_ = torch::tensor(at::ArrayRef<uint8_t>(cv_rgb.data,cv_rgb.rows*cv_rgb.cols*3)).view({cv_rgb.rows,cv_rgb.cols,3});
  //    auto label_ = torch::tensor(at::ArrayRef<float>((float*)cv_depth_meters.data,
  //    cv_depth_meters.rows*cv_depth_meters.cols)).view({cv_depth_meters.rows,cv_depth_meters.cols,1});
  //
  //    // Note that opencv stores things in [rgb,row,col] so we need to flip it after loading it into the tensor
  //    // Our view is of the "outer most" dimension, so we start with the number of columns, then rows, then the rgb
  //    input_ = input_.permute({2,0,1}).clone();
  //    label_ = label_.permute({2,0,1}).clone();
  //
  //    // Convert our tensors to float and long types
  //    // Also we scale our input rbg image to be between [0..1]
  //    input_ = input_.to(torch::kFloat);
  //    input_ = input_/255.0;
  //    label_ = label_.to(torch::kFloat);
  //
  //    // Return our input and target
  //    return torch::data::Example<>(input_,label_);
}
