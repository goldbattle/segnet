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

#ifndef UNETMODEL_H
#define UNETMODEL_H

#include <torch/nn/modules.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

#include "network/blocks/UNetBlocks.h"

/**
 * This is the implementation class for the UNet network
 * Here we construct all the layers, and also have the feed forward function
 * Pytorch will handle the autograd computation for us
 * Output is a N channel probability that each pixel is a given class
 */
struct UNetModelImpl : torch::nn::Module {

  /**
   * Default constructor
   * Should initialize all of our sub-modules
   */
  UNetModelImpl(size_t n_channels, size_t n_classes) {

    // NOTE: We need to ensure that all modules have been
    // NOTE: Constructed using the "empty holder" by passing a nullptr for their
    // NOTE: Default constructor, see the bottom of this class
    // NOTE: Because theses modules are sharedptrs, we need to make sure we are referencing

    // Input convolutions
    inconv1 = register_module("inconv1", UNetConvolution(n_channels, 64));
    inconv2 = register_module("inconv2", UNetConvolution(64, 64));

    // Downscaling layers
    down1 = register_module("down1", UNetDownwards(64, 128));
    down2 = register_module("down2", UNetDownwards(128, 256));
    down3 = register_module("down3", UNetDownwards(256, 512));
    down4 = register_module("down4", UNetDownwards(512, 512));

    // Original UNET architecture, don't use since it takes up a lot of memory
    // Note we will be appending the downsample layers, so we add their amount to the input size
    // down4 = register_module("down4", UNetDownwards(512,1024));
    // up1 = register_module("up1", UNetUpwards(1024+512,256));

    // Upscaling layers
    up1 = register_module("up1", UNetUpwards(1024, 256));
    up2 = register_module("up2", UNetUpwards(512, 128));
    up3 = register_module("up3", UNetUpwards(256, 64));
    up4 = register_module("up4", UNetUpwards(128, 64));

    // Final convolution layer
    outconv = register_module("outconv", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, n_classes, 3).padding(1)));
  }

  /**
   * Network forward propagation
   * INPUT: image
   * OUTPUT: N masks for each class
   */
  torch::Tensor forward(torch::Tensor input) {

    // Final output of the network
    torch::Tensor output;

    // Outputs of our downsampling layers
    torch::Tensor x1, x2, x3, x4, x5;

    // First do our starting two convolutions
    x1 = inconv1(input);
    x1 = inconv2(x1);
    x1 = torch::dropout(x1, 0.25, this->is_training());

    // Downscale to the bottleneck
    x2 = down1(x1);
    x3 = down2(x2);
    x4 = down3(x3);
    x5 = down4(x4);

    // Upsample and concatenate downlayers
    output = up1(x5, x4);
    output = up2(output, x3);
    output = up3(output, x2);
    output = up4(output, x1);

    // Finally last convolution and return
    return outconv(output);
  }

  // NOTE: Parts of the network
  // NOTE: For submodules, we call the "empty holder" constructor
  // NOTE: https://pytorch.org/tutorials/advanced/cpp_frontend.html

  // First set of convolutions
  UNetConvolution inconv1{nullptr};
  UNetConvolution inconv2{nullptr};

  // Down segments (maxpool => 2x conv)
  UNetDownwards down1{nullptr};
  UNetDownwards down2{nullptr};
  UNetDownwards down3{nullptr};
  UNetDownwards down4{nullptr};

  // Upscaling sections (upscale => cat => conv)
  UNetUpwards up1{nullptr};
  UNetUpwards up2{nullptr};
  UNetUpwards up3{nullptr};
  UNetUpwards up4{nullptr};

  // Last convolution for the final layer
  torch::nn::Conv2d outconv{nullptr};
};

// Now actually create the UNetModel object
// This macro will use the above implementation
// And construct a sharedptr for this class
// (so we are always passing by reference throughout the program)
TORCH_MODULE(UNetModel);

#endif // UNETMODEL_H