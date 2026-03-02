#ifndef MODELTOOLS3D_H
#define MODELTOOLS3D_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <cmath>

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

// Platform-specific SIMD
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define USE_SSE_3D 1
    #include <xmmintrin.h>
#elif defined(__aarch64__) || defined(__arm__)
    #define USE_NEON_3D 1
    #include <arm_neon.h>
    typedef float32x4_t __m128;
    #define _mm_setzero_ps()       vdupq_n_f32(0.0f)
    #define _mm_set_ps(a,b,c,d)    ((float32x4_t){d,c,b,a})
    #define _mm_load_ps(p)         vld1q_f32(p)
    #define _mm_mul_ps(a,b)        vmulq_f32(a,b)
    #define _mm_add_ps(a,b)        vaddq_f32(a,b)
    #define _mm_max_ps(a,b)        vmaxq_f32(a,b)
    inline void* _mm_malloc(size_t sz, size_t al) { void* p=nullptr; posix_memalign(&p,al,sz); return p; }
    inline void  _mm_free(void* p) { free(p); }
#else
    #define USE_SCALAR_3D 1
    inline void* _mm_malloc(size_t sz, size_t al) { void* p=nullptr; posix_memalign(&p,al,sz); return p; }
    inline void  _mm_free(void* p) { free(p); }
#endif

#include "./Constants.h"
#include "./Point.h"

using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace std;
using namespace rsmi3dent;

namespace rsmi3dutil {

// ---------------------------------------------------------------------------
// Net3D – same 2-layer MLP as RSMI, but accepts 3-D input (x,y,z).
// input_width MUST be 3.
// ---------------------------------------------------------------------------
struct Net : torch::nn::Module
{
public:
    int input_width = 3;
    int max_error = 0;
    int min_error = 0;
    int width = 0;

    float learning_rate = Constants::LEARNING_RATE;

    // fc1 weight matrix, flattened row-major: [w_h0_x, w_h0_y, w_h0_z, w_h1_x, ...]
    float  w1[Constants::HIDDEN_LAYER_WIDTH * 3];
    float  w2[Constants::HIDDEN_LAYER_WIDTH];
    float  b1[Constants::HIDDEN_LAYER_WIDTH];
    float  b2 = 0.0f;

    // SIMD-friendly column-major weight arrays (one array per input dimension)
    float *w1_0 = (float*)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *w1_1 = (float*)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *w1_2 = (float*)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *w2_  = (float*)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);
    float *b1_  = (float*)_mm_malloc(Constants::HIDDEN_LAYER_WIDTH * sizeof(float), 32);

    Net(int in_width, int hidden_width)
    {
        this->input_width = in_width;
        this->width = hidden_width >= Constants::HIDDEN_LAYER_WIDTH
                      ? Constants::HIDDEN_LAYER_WIDTH : hidden_width;
        fc1 = register_module("fc1", torch::nn::Linear(input_width, this->width));
        fc2 = register_module("fc2", torch::nn::Linear(this->width, 1));
        torch::nn::init::uniform_(fc1->weight, 0, 0.1);
        torch::nn::init::uniform_(fc2->weight, 0, 0.1);
    }

    explicit Net(int in_width)
    {
        this->input_width = in_width;
        this->width = Constants::HIDDEN_LAYER_WIDTH;
        fc1 = register_module("fc1", torch::nn::Linear(input_width, this->width));
        fc2 = register_module("fc2", torch::nn::Linear(this->width, 1));
        torch::nn::init::uniform_(fc1->weight, 0, 0.1);
        torch::nn::init::uniform_(fc2->weight, 0, 0.1);
    }

    // Extract fc1 weights into column-major SIMD arrays.
    // PyTorch stores fc1->weight as shape [width, input_width] row-major.
    // Row h = [w_h_x, w_h_y, w_h_z] → after flatten: index 3h, 3h+1, 3h+2.
    void get_parameters()
    {
        torch::Tensor p1 = this->parameters()[0];   // fc1 weight
        torch::Tensor p2 = this->parameters()[1];   // fc1 bias
        torch::Tensor p3 = this->parameters()[2];   // fc2 weight
        torch::Tensor p4 = this->parameters()[3];   // fc2 bias

        p1 = p1.reshape({3 * width, 1});
        for (int i = 0; i < width; i++) {
            w1[i*3]   = p1.select(0, 3*i  ).item().toFloat();
            w1[i*3+1] = p1.select(0, 3*i+1).item().toFloat();
            w1[i*3+2] = p1.select(0, 3*i+2).item().toFloat();
            w1_0[i]   = w1[i*3  ];
            w1_1[i]   = w1[i*3+1];
            w1_2[i]   = w1[i*3+2];
        }
        p2 = p2.reshape({width, 1});
        for (int i = 0; i < width; i++) {
            b1[i]  = p2.select(0, i).item().toFloat();
            b1_[i] = b1[i];
        }
        p3 = p3.reshape({width, 1});
        for (int i = 0; i < width; i++) {
            w2[i]  = p3.select(0, i).item().toFloat();
            w2_[i] = w2[i];
        }
        b2 = p4.item().toFloat();
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::Tensor predict(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    float activation(float val) { return val > 0.0f ? val : 0.0f; }

    // SIMD fast inference for a single 3-D point.
    float predict(Point point)
    {
        float px = point.x, py = point.y, pz = point.z;
        int blocks   = width / 4;
        int rem      = width % 4;
        int move_back = blocks * 4;

        float result = 0.0f;

#if defined(USE_SSE_3D) || defined(USE_NEON_3D)
        __m128 fSum0       = _mm_setzero_ps();
        __m128 fLoad0_x    = _mm_set_ps(px, px, px, px);
        __m128 fLoad0_y    = _mm_set_ps(py, py, py, py);
        __m128 fLoad0_z    = _mm_set_ps(pz, pz, pz, pz);
        __m128 fLoad0_zero = _mm_setzero_ps();

        for (int i = 0; i < blocks; i++) {
            __m128 fw1_x = _mm_load_ps(w1_0);
            __m128 fw1_y = _mm_load_ps(w1_1);
            __m128 fw1_z = _mm_load_ps(w1_2);
            __m128 fb1   = _mm_load_ps(b1_);
            __m128 fw2   = _mm_load_ps(w2_);

            __m128 t = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(fLoad0_x, fw1_x),
                                   _mm_mul_ps(fLoad0_y, fw1_y)),
                        _mm_add_ps(_mm_mul_ps(fLoad0_z, fw1_z), fb1));
            t = _mm_max_ps(t, fLoad0_zero);           // ReLU
            fSum0 = _mm_add_ps(fSum0, _mm_mul_ps(t, fw2));

            w1_0 += 4; w1_1 += 4; w1_2 += 4; b1_ += 4; w2_ += 4;
        }
        if (blocks > 0)
            result += fSum0[0] + fSum0[1] + fSum0[2] + fSum0[3];
#endif

        for (int i = 0; i < rem; i++) {
            result += activation(px * w1_0[i] + py * w1_1[i] + pz * w1_2[i] + b1_[i]) * w2_[i];
        }
        result += b2;

        // Reset sliding pointers
        w1_0 -= move_back; w1_1 -= move_back; w1_2 -= move_back;
        b1_  -= move_back; w2_  -= move_back;
        return result;
    }

    void train_model(vector<float> locations, vector<float> labels)
    {
        long long N = labels.size();
#ifdef use_gpu
        torch::Tensor x = torch::tensor(locations, at::kCUDA).reshape({N, input_width});
        torch::Tensor y = torch::tensor(labels, at::kCUDA).reshape({N, 1});
#else
        torch::Tensor x = torch::tensor(locations).reshape({N, input_width});
        torch::Tensor y = torch::tensor(labels).reshape({N, 1});
#endif
        torch::optim::Adam optimizer(this->parameters(),
                                     torch::optim::AdamOptions(learning_rate));
        for (int epoch = 0; epoch < Constants::EPOCH; epoch++) {
            optimizer.zero_grad();
            torch::Tensor loss = torch::mse_loss(this->forward(x), y);
            loss.backward();
            optimizer.step();
        }
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

}

#endif
