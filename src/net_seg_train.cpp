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

int main() {

  // See if we can use the GPU or should train using the CPU
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU. Are you using the CUDA TorchLib??" << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // Load the data from disk
  std::string rootdir = "/media/patrick/DATA 02/SEGNET/comma10k/";
  // auto dataset =
  //     CityscapesDataset(rootdir, CityscapesDataset::ModeDataSplit::kTrain, CityscapesDataset::ModeGroundtruth::kSegmentation, true);
  auto dataset = Comma10kDataset(rootdir, Comma10kDataset::ModeDataSplit::kTrain, true);

  // Create the network (RGB image => class masks)
  size_t n_channels = 3;
  size_t n_classes = dataset.map_id2class.size();
  UNetModel model(n_channels, n_classes);
  std::cout << "num params: " << model->parameters(true).size() << std::endl;

  // Load from disk if we can
  std::string modelpath = "checkpoint-seg.pt";
  if (boost::filesystem::exists(modelpath)) {
    torch::load(model, modelpath);
  } else {
    std::cerr << "no checkpoint found on disk, creating new model." << std::endl;
  }
  model->to(device);

  // Create the log file for our loss values
  std::string logpath = "loss_history_seg.txt";
  std::ofstream loghist;
  loghist.open(logpath, std::ofstream::out | std::ofstream::app);

  // Finally convert it to a unique pointer dataloader
  auto dataset_mapped = dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset_mapped), torch::data::DataLoaderOptions().batch_size(5).workers(30));

  // Create the optimizer
  // torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.0002));

  // Loop through our epochs
  size_t max_epochs = 100;
  for (size_t epoch = 1; epoch <= max_epochs; epoch++) {

    // Enable training mode
    model->train();

    // Loop through our batches of training data
    size_t batch_idx = 0;
    for (auto &batch : *data_loader) {

      // Send that data to the device
      auto d_rgb = batch.data.to(device);
      auto d_masks = batch.target.to(device);

      // Zero the gradient from the last run
      optimizer.zero_grad();

      // Send the data through and calc the loss
      auto output = model->forward(d_rgb);

      // There is not a cross entropy loss
      // We do a softmax first along the classes [N, classes, H, W]
      // Also note that our dataloader will add an extra dimension to the labels (so we squeese the class dim)
      // https://discuss.pytorch.org/t/c-loss-functions/27471/5
      d_masks = torch::squeeze(d_masks, 1);
      auto loss = torch::nll_loss2d(torch::log_softmax(output, 1), d_masks);
      loghist << std::fixed << std::setprecision(10) << loss.item<float>() << std::endl;

      // Back propagate!
      loss.backward();
      optimizer.step();

      // Print our the loss every once in a while
      if (batch_idx % 10 == 0) {

        // Debug printout
        size_t items_curr = batch_idx * batch.data.size(0);
        size_t items_total = dataset.size().value();
        std::cout << epoch << " - " << items_curr << "/" << items_total << " | loss = " << loss.item<float>() << std::endl;

        // Softmax the output to get our total class probabilities
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

        // Finally, stack and display in a window
        cv::Mat outimg1, outimg2, outimg3;
        cv::hconcat(img_input, img_label, outimg1);
        cv::hconcat(img_input, img_output, outimg2);
        cv::vconcat(outimg1, outimg2, outimg3);
        cv::imshow("current status", outimg3);
        cv::waitKey(100);

        // Save the model to disk
        torch::save(model, modelpath);
      }
      batch_idx++;
    }
  }

  // Save model to file
  torch::save(model, modelpath);

  // Close file
  loghist.close();
}
