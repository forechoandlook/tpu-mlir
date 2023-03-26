//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PermuteOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  i32_array_t in_order = module::getI32Array(getOrder());
  auto input_addr = module::getAddress(input);
  auto output_addr = module::getAddress(output);
  uint64_t buffer_addr = module::getAddress(getBuffer());

  // melloc
  uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
  int *order = new int[MAX_SHAPE_DIMS];

  for (auto v : llvm::enumerate(module::getShape(input)))
    input_shape[v.index()] = (uint32_t)v.value();
  memcpy(order, in_order->data(), (*in_order).size() * sizeof(int));

  int input_dims = module::getShape(input).size();
  auto input_dtype = BM1684::getDataType(input);
  auto output_dtype = BM1684::getDataType(output);
  int store_mode;
  if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
      input_dtype == DTYPE_UINT32) {
    store_mode = STORE_MODE_1N;
    BM1684::instance().dl_nodechip_transpose(
        input_addr, output_addr, input_shape, order, input_dims, sizeof(float),
        store_mode, buffer_addr, NULL,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    store_mode = STORE_MODE_4N;
    assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
    BM1684::instance().dl_nodechip_transpose_fix8b(
        input_addr, output_addr, input_shape, order, input_dims, store_mode,
        store_mode, buffer_addr, NULL,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    llvm_unreachable("Not Implemented");
  }

  // release
  delete[] input_shape;
  delete[] order;
}
