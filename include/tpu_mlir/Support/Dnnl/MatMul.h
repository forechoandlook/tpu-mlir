//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace tpu_mlir {
class MatMul {
public:
  MatMul();

  void right_init(float *right, int64_t right_zp, int64_t batch, int64_t K,
                  int64_t N, bool right_transpose, bool hdim_is_batch,
                  int64_t batch_low);
  void input_init(float *input, int64_t input_zp, int64_t batch, int64_t M,
                  int64_t K);
  void setup(float *left, float *right, float *bias, float *output,
             int64_t batch, int64_t M, int64_t K, int64_t N, bool do_relu,
             double relu_limit, int64_t right_zp, bool right_transpose,
             int64_t input_zp, bool hdim_is_batch = false,
             int64_t batch_low = 1);

  void run();

private:
  engine eng;
  stream engine_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  std::shared_ptr<std::vector<float>> bias0;
  float *p_right, *p_input;
  float *origin_input, *origin_right;
  std::shared_ptr<std::vector<float>> right_after_init;
  std::shared_ptr<std::vector<float>> input_after_init;
  int64_t batch_, M_, N_, K_, right_zp_, input_zp_;
  bool right_has_zp_ = 0, input_has_zp_ = 0, has_transpose_ = 0;
  bool hdim_is_batch_;
  int64_t batch_low_;
};
} // namespace tpu_mlir
