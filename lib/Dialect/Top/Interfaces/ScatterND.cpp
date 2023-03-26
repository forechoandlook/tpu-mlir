//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::ScatterNDOp::getFLOPs() { return 0; }

LogicalResult top::ScatterNDOp::init(InferenceParameter &p) {
  return success();
}
void top::ScatterNDOp::deinit(InferenceParameter &p) {}

LogicalResult top::ScatterNDOp::inference(InferenceParameter &p) {
  const float *data = p.inputs[0];
  const float *indices = p.inputs[1];
  const float *updates = p.inputs[2];
  float *dst = p.outputs[0];

  auto data_shape = module::getShape(getInputData());
  auto indices_shape = module::getShape(getIndices());
  auto updates_shape = module::getShape(getUpdates());
  auto dtype_size = module::getDtypeSize(getInputData());
  int r = data_shape.size();
  int q = indices_shape.size();
  int k = indices_shape[q - 1];
  int updates_dims = sizeof(updates_shape) / sizeof(updates_shape[0]);
  assert(updates_dims == q + r - k - 1);
  int updates_elems = 1;
  int slice_elems = 1;
  for (int i = 0; i < q - 1; ++i) {
    assert(updates_shape[i] == indices_shape[i]);
    updates_elems *= indices_shape[i];
  }
  for (int j = 0; j < r - k; ++j) {
    assert(updates_shape[q - 1 + j] == data_shape[k - 1 + j]);
    slice_elems *= updates_shape[q - 1 + j];
  }
  auto data_elems = module::getNumElements(getOutput());

  memcpy(dst, data, data_elems * dtype_size);

  int data_strides[k];

  for (int dim = k - 1; dim >= 0; --dim) {
    if (dim == k - 1)
      data_strides[dim] = 1;
    else
      data_strides[dim] = data_strides[dim + 1] * data_shape[dim + 1];
  }
  int64_t idx = 0;
  if (r == k) {
    // #pragma omp parallel for schedule(static, omp_schedule(updates_elems))
    for (int64_t loc = 0; loc < updates_elems; ++loc) {
      idx = 0;
      for (int64_t i = 0; i < k; ++i) {
        idx += indices[loc * k + i] * data_strides[i];
      }
      dst[idx] = updates[loc];
    }
  } else if (k < r) {
    // #pragma omp parallel for schedule(static, omp_schedule(updates_elems))
    for (int64_t loc = 0; loc < updates_elems; ++loc) {
      idx = 0;
      for (int64_t i = 0; i < k; ++i) {
        idx += indices[loc * k + i] * data_strides[i];
      }
      memcpy(dst + idx, updates + loc, slice_elems);
    }
  }

  return success();
}

void top::ScatterNDOp::shape_inference() {}
