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

#include <boost/filesystem.hpp>
#include <iostream>
#include <torch/data.h>
#include <torch/torch.h>

#include "data/CityscapesDataset.h"
#include "data/Comma10kDataset.h"
#include "network/models/UNetModel.h"

int main(int argc, char *argv[]) {

  // See if we can use the GPU or should train using the CPU
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Predicting on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Predicting on CPU. Are you using the CUDA TorchLib??" << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // Get the checkpoint if passed
  std::string modelpath = "checkpoint-seg.pt";
  std::string rootdir = "/media/patrick/DATA 02/SEGNET/comma10k/";
  if (argc == 2) {
    modelpath = std::string(argv[1]);
  } else if (argc == 3) {
    modelpath = std::string(argv[1]);
    rootdir = std::string(argv[2]);
  }

  // Load the data from disk
  auto dataset = Comma10kDataset(rootdir, Comma10kDataset::ModeDataSplit::kTest, false);

  // Create the network (RGB image => class masks)
  size_t n_channels = 3;
  size_t n_classes = dataset.map_id2class.size();
  UNetModel model(n_channels, n_classes);
  std::cout << "num params: " << model->parameters(true).size() << std::endl;

  // Load from disk if we can
  if (boost::filesystem::exists(modelpath)) {
    torch::load(model, modelpath);
  } else {
    std::cerr << "no checkpoint found on disk, creating new model." << std::endl;
  }
  model->to(device);

  // Set that our model will not need the gradients
  // Reduces computation by 1-2ms in total
  torch::NoGradGuard no_grad;
  model->eval();

  // Finally convert it to a unique pointer dataloader
  auto dataset_mapped = dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset_mapped), torch::data::DataLoaderOptions().batch_size(1).workers(6));

  // Loop through our batches of training data
  double loss_sum = 0.0;
  size_t loss_ct = 0;
  size_t batch_idx = 0;
  for (auto &batch : *data_loader) {

    // Send that data to the device
    auto d_rgb = batch.data.to(device);
    auto d_masks = batch.target.to(device);

    // Send the data through and calc the loss
    auto output = model->forward(d_rgb);

    // There is not a cross entropy loss
    // We do a softmax first along the classes [N, classes, H, W]
    // Also note that our dataloader will add an extra dimension to the labels (so we squeese the class dim)
    // https://discuss.pytorch.org/t/c-loss-functions/27471/5
    d_masks = torch::squeeze(d_masks, 1);
    auto loss = torch::nll_loss2d(torch::log_softmax(output, 1), d_masks);
    loss_sum += loss.item<float>();
    loss_ct += batch.data.size(0);

    // Debug printout
    size_t items_curr = batch_idx * batch.data.size(0);
    size_t items_total = dataset.size().value();
    double loss_avg = loss_sum / (double)loss_ct;
    std::cout << items_curr << "/" << items_total << " | loss = " << loss.item<float>() << " | loss_avg = " << loss_avg << " (" << loss_ct
              << " samples)" << std::endl;

    // Softmax the output to get our total class probabilities [N, classes, H, W]
    // Thus across all classes, our probabilities should sum to 1
    auto output_probs = torch::softmax(output, 1);

    // Plot the first image, need to change to opencv format [H,W,C]
    // Note that we arg max the softmax network output, then need to add an dimension
    // We scale up the 0..1 range back to the 0..255 that opencv expects (later cast to int)
    torch::Tensor cv_input = 255.0 * batch.data[0].permute({1, 2, 0}).clone().cpu();
    torch::Tensor cv_label = batch.target[0].permute({1, 2, 0}).clone().cpu();
    torch::Tensor cv_output = torch::unsqueeze(output_probs[0].argmax(0), 0).permute({1, 2, 0}).clone().cpu();

    // Convert them all to 0..255 ranges
    cv_input = cv_input.to(torch::kInt8);
    cv_label = cv_label.to(torch::kInt8);
    cv_output = cv_output.to(torch::kInt8);

    // Point the cv::Mats to the transformed locations in memory
    cv::Mat img_input(cv::Size((int)cv_input.size(1), (int)cv_input.size(0)), CV_8UC3, cv_input.data_ptr<int8_t>());
    cv::Mat img_label(cv::Size((int)cv_label.size(1), (int)cv_label.size(0)), CV_8UC1, cv_label.data_ptr<int8_t>());
    cv::Mat img_output(cv::Size((int)cv_output.size(1), (int)cv_output.size(0)), CV_8UC1, cv_output.data_ptr<int8_t>());

    // Convert labeled images to color
    cv::cvtColor(img_label, img_label, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img_output, img_output, cv::COLOR_GRAY2BGR);
    // img_label = 255.0 / (double)n_classes * img_label;
    // img_output = 255.0 / (double)n_classes * img_output;

    // Change both to be colored like the comma10k
    img_label.forEach<cv::Vec3b>([&](cv::Vec3b &px, const int *pos) -> void { px = dataset.map_id2hex[(char)px[0]]; });
    img_output.forEach<cv::Vec3b>([&](cv::Vec3b &px, const int *pos) -> void { px = dataset.map_id2hex[(char)px[0]]; });

    // Finally stack and display in a window
    cv::Mat outimg1, outimg2, outimg3;
    cv::hconcat(img_input, img_label, outimg1);
    cv::hconcat(img_input, img_output, outimg2);
    cv::vconcat(outimg1, outimg2, outimg3);
    cv::imshow("prediction", outimg3);

    // Next we will visualize our probability distributions  [N, classes, H, W]
    torch::Tensor cv_probs = output_probs[0].clone().cpu();
    cv_probs = cv_probs.to(torch::kFloat32);
    cv::Mat outimg4 = cv::Mat(cv::Size(n_classes * (int)cv_input.size(1), (int)cv_input.size(0)), CV_8UC3, cv::Scalar(0, 0, 0));
    assert((size_t)output_probs.size(0) == 1);
    assert((size_t)cv_probs.size(0) == n_classes);
    for (int n = 0; n < (int)n_classes; n++) {
      cv::Mat imgtmp(cv::Size((int)cv_probs.size(2), (int)cv_probs.size(1)), CV_32FC1, cv_probs[n].data_ptr<float>());
      imgtmp = 255 * imgtmp;
      imgtmp.convertTo(imgtmp, CV_8UC1);
      cv::Mat imgtmp_color;
      cv::applyColorMap(imgtmp, imgtmp_color, cv::COLORMAP_JET);
      imgtmp_color.copyTo(outimg4(cv::Rect(n * (int)cv_input.size(1), 0, imgtmp.cols, imgtmp.rows)));
    }
    cv::imshow("uncertainties", outimg4);
    cv::waitKey(100);

    // Save to file for readme
    // cv::imwrite("/home/patrick/github/segnet/docs/example_pred.png", outimg3);
    // cv::imwrite("/home/patrick/github/segnet/docs/example_probs.png", outimg4);
    // std::exit(EXIT_FAILURE);

    batch_idx++;
  }
}
