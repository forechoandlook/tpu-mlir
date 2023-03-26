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
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "detection-output"



int64_t top::DetectionOutputOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::DetectionOutputOp::init(InferenceParameter &p) {
  return success();
}
void top::DetectionOutputOp::deinit(InferenceParameter &p) {}

LogicalResult top::DetectionOutputOp::inference(InferenceParameter &p) {
  DetParam param;
  param.keep_top_k = this->getKeepTopK();
  param.confidence_threshold = this->getConfidenceThreshold().convertToDouble();
  param.nms_threshold = this->getNmsThreshold().convertToDouble();
  param.top_k = this->getTopK();
  param.num_classes = this->getNumClasses();
  param.share_location = this->getShareLocation();
  param.background_label_id = this->getBackgroundLabelId();
  param.loc_shape = module::getShape(this->getInputs()[0]);
  param.conf_shape = module::getShape(this->getInputs()[1]);
  //onnx ssd just have loc、conf
  if (this->getInputs().size() >= 3) {
    param.prior_shape = module::getShape(this->getInputs()[2]);
    param.onnx_nms = 0;
  } else {
    param.onnx_nms = 1;
  }

  std::string str_code_type = this->getCodeType().str();
  if (str_code_type == "CORNER") {
    param.code_type = PriorBoxParameter_CodeType_CORNER;
  } else if (str_code_type == "CENTER_SIZE") {
    param.code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  } else if (str_code_type == "CORNER_SIZE") {
    param.code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
  } else {
    llvm_unreachable("code type wrong");
  }

  param.loc_data = p.inputs[0];
  param.conf_data = p.inputs[1];
  param.prior_data = param.onnx_nms ? nullptr :p.inputs[2];
  param.output_data = p.outputs[0];

  DetectionOutputFunc det_func(param);
  det_func.invoke();
  return success();
}

void top::DetectionOutputOp::shape_inference() {}
