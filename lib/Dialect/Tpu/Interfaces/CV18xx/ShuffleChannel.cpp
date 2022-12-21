//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// #include "tpu_mlir/Backend/BM168x/cv18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::ShuffleChannelOp::codegen_global_cv18xx( int64_t layer_id) {
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  std::vector<int64_t> input_shape;
  Module::getShapeVec(input(), input_shape);
  int64_t group = this->group();
  if (Quant::isUniformQuantized(output())) {
    cvi_backend_tg_permute_kernel( layer_id, ga_input, ga_output,
          input_shape[0], group, input_shape[1] / group, input_shape[2] * input_shape[3],
          0, 2, 1, 3, CVK_FMT_I8);
  } else {
    cvi_backend_tg_permute_kernel( layer_id, ga_input, ga_output,
          input_shape[0], group, input_shape[1] / group, input_shape[2] * input_shape[3],
          0, 2, 1, 3, CVK_FMT_BF16);
  }
}