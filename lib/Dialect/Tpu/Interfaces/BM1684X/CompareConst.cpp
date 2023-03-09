//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================

void tpu::CompareConstOp::codegen_global_bm1684x() {
  constbinary_global_spec_t spec = {0};
  spec.common.B_const_val = getConstVal().convertToDouble();
  spec.common.inversed = getInversed();
  spec.common.binary_type = BM168x::compare_mode(getMode());
  spec.common.if_relu = 0;
  spec.common.rshift_A = 0;
  spec.common.scale_A = 1;
  spec.common.B_dtype = BM168x::getDataType(getInput());
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_constbinary_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CompareConstOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice,
    group_type_t group_type) {
  return 0;
}

void tpu::CompareConstOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                                group_type_t group_type,
                                                local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  constbinary_local_spec_t spec = {0};
  spec.common.B_const_val = getConstVal().convertToDouble();
  spec.common.inversed = getInversed();
  spec.common.binary_type = BM168x::compare_mode(getMode());
  spec.common.if_relu = 0;
  spec.common.rshift_A = 0;
  spec.common.scale_A = 1;
  spec.common.B_dtype = BM168x::getDataType(getInput());

  BM168x::call_local_func("backend_api_constbinary_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::CompareConstOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(constbinary_local_param_t);
  constbinary_local_param_t param = {0};
  param.spec.common.B_const_val = getConstVal().convertToDouble();
  param.spec.common.inversed = getInversed();
  param.spec.common.binary_type = BM168x::compare_mode(getMode());
  param.spec.common.if_relu = 0;
  param.spec.common.rshift_A = 0;
  param.spec.common.scale_A = 1;
  param.spec.common.B_dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CompareConstOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(constbinary_global_spec_t);
  constbinary_global_spec_t spec = {0};
  spec.common.B_const_val = getConstVal().convertToDouble();
  spec.common.inversed = getInversed();
  spec.common.binary_type = BM168x::compare_mode(getMode());
  spec.common.if_relu = 0;
  spec.common.rshift_A = 0;
  spec.common.scale_A = 1;
  spec.common.B_dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::CompareConstOp::get_layer_type() {
  return FW_BMNET_CONST_BINARY;
}
