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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::SqueezeOp::getFLOPs() { return 0; }

LogicalResult top::SqueezeOp::init(InferenceParameter &p) { return success(); }
void top::SqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult top::SqueezeOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; i++) {
    p.outputs[0][i] = p.inputs[0][i];
  }
  return success();
}

void top::SqueezeOp::shape_inference() {
  auto in_shape = module::getShape(getInputs());
  auto axes = module::getI64Array(getAxesAttr());
  std::vector<int64_t> out_shape;
  int64_t in_dims = in_shape.size();
  for (int i = 0; i < in_dims; ++i) {
    out_shape.push_back(in_shape[i]);
    for (auto axis : *axes) {
      if (axis < 0) {
        axis += in_dims;
      }
      if (axis == i) {
        out_shape.pop_back();
      }
    }
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  // common_shape_inference(getOperation());
}
