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

#ifndef UNETBLOCKS_H
#define UNETBLOCKS_H

#include <torch/nn/pimpl.h>
#include <torch/torch.h>

/**
 * UNet convolution layer
 * LAYER: (conv => BN => ReLU)
 */
struct UNetConvolutionImpl : torch::nn::Module {

  // Default constructor
  UNetConvolutionImpl(size_t n_in, size_t n_out) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(n_in, n_out, 3).padding(1)));
    bn = register_module("bn", torch::nn::BatchNorm(torch::nn::BatchNormOptions(n_out)));
  }

  // Forward propagation
  torch::Tensor forward(torch::Tensor input) { return torch::relu(bn(conv(input))); }

  // Parts of the network
  // NOTE: for submodules, we call the "empty holder" constructor
  // NOTE: https://pytorch.org/tutorials/advanced/cpp_frontend.html
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm bn{nullptr};
};
TORCH_MODULE(UNetConvolution);

/**
 * UNet downscaling layers
 * LAYER: maxpool => (conv => BN => ReLU) * 2
 */
struct UNetDownwardsImpl : torch::nn::Module {

  // Default constructor
  UNetDownwardsImpl(size_t n_in, size_t n_out) {
    conv1 = register_module("conv1", UNetConvolution(n_in, n_out));
    conv2 = register_module("conv2", UNetConvolution(n_out, n_out));
  }

  // Forward propagation
  torch::Tensor forward(torch::Tensor input) { return conv2(conv1(torch::max_pool2d(input, 2))); }

  // Parts of the network
  // NOTE: for submodules, we call the "empty holder" constructor
  // NOTE: https://pytorch.org/tutorials/advanced/cpp_frontend.html
  UNetConvolution conv1{nullptr};
  UNetConvolution conv2{nullptr};
};
TORCH_MODULE(UNetDownwards);

/**
 * UNet upsampling layers
 * LAYER: upscale => cat => (conv => BN => ReLU) * 2
 */
struct UNetUpwardsImpl : torch::nn::Module {

  // Default constructor
  UNetUpwardsImpl(size_t n_in, size_t n_out) {
    conv1 = register_module("conv1", UNetConvolution(n_in, n_out));
    conv2 = register_module("conv2", UNetConvolution(n_out, n_out));
  }

  // Forward propagation
  torch::Tensor forward(torch::Tensor input, torch::Tensor bridge) {

    // First upsample the input (tensors are [N, channels, H, W])
    // input = torch::upsample_bilinear2d(input,{bridge.size(2),bridge.size(3)},true);
    input = torch::upsample_bicubic2d(input, {bridge.size(2), bridge.size(3)}, true);

    // Concatenate along the "channel" direction
    input = torch::cat({input, bridge}, 1);

    // Finally do our convolutions and return
    return conv2(conv1(input));
  }

  // Parts of the network
  // NOTE: for submodules, we call the "empty holder" constructor
  // NOTE: https://pytorch.org/tutorials/advanced/cpp_frontend.html
  UNetConvolution conv1{nullptr};
  UNetConvolution conv2{nullptr};
};
TORCH_MODULE(UNetUpwards);

#endif // UNETBLOCKS_H