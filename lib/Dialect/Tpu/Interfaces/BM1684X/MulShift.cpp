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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;


void tpu::MulShiftOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  BM168x::getBetterNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    auto in_zp = in_qtype.getZeroPoint();
    auto out_zp = out_qtype.getZeroPoint();
    if (in_zp != 0 || out_zp != 0) {
      requant_int_param_t param = {0};
      param.input_addr = module::getAddress(getInput());
      param.output_addr = module::getAddress(getOutput());
      param.n = (int)n;
      param.c = (int)c;
      param.h = (int)h;
      param.w = (int)w;
      param.mul_value = getMultiplier();
      param.shift_value = -getRshift();
      param.offset_value = out_zp;
      param.zx_value = in_zp;
      param.mode = 2;
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      BM168x::call_global_func("backend_api_requant_int_global", &param,
                               sizeof(param));
      return;
    }
  }
  mulshift_param_t param = {0};
  param.input_addr = module::getAddress(getInput());
  param.output_addr = module::getAddress(getOutput());
  param.input_n = n;
  param.input_c = c;
  param.input_h = h;
  param.input_w = w;
  param.scale_val = getMultiplier();
  param.rshift_num = getRshift();
  param.input_dtype = BM168x::getDataType(getInput());
  param.scale_dtype = param.scale_val < 0 ? DTYPE_INT8 : DTYPE_UINT8;
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = ROUND_UP;
  BM168x::call_global_func("backend_api_mulshift_global", &param,
                           sizeof(param));
}

int64_t tpu::MulShiftOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice,
    group_type_t group_type) {
  auto in_sType = module::getStorageType(getInput());
  auto out_sType = module::getStorageType(getOutput());
  if (module::isUniformQuantized(getInput())) {
    auto in_qType = module::getUniformQuantizedType(getInput());
    auto out_qType = module::getUniformQuantizedType(getOutput());
    if (in_qType.getZeroPoint() != 0 || out_qType.getZeroPoint() != 0) {
      return 2 * in_lmem_bytes;
    }
  }
  if (in_sType.isUnsignedInteger(8) == false &&
      out_sType.isUnsignedInteger(8)) {
    return 2 * in_lmem_bytes;
  }
  return 0;
}

void tpu::MulShiftOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w, group_type);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);

  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    auto in_zp = in_qtype.getZeroPoint();
    auto out_zp = out_qtype.getZeroPoint();
    if (in_zp != 0 || out_zp != 0) {
      requant_int_param_t param = {0};
      param.input_addr = (uint32_t)in_gi.out_addr;
      param.output_addr = (uint32_t)gi.out_addr;
      param.buffer_local_addr = (uint32_t)gi.buffer_addr;
      param.n = sec_info.out_n_slice;
      param.c = c;
      param.h = sec_info.out_h_slice;
      param.w = sec_info.out_w_slice;
      param.mul_value = getMultiplier();
      param.shift_value = -getRshift();
      param.offset_value = out_zp;
      param.zx_value = in_zp;
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      param.mode = 2;
      BM168x::call_local_func("backend_api_requant_int_local", &param,
                              sizeof(param));
      return;
    }
  }
  mulshift_param_t param = {0};
  param.input_addr = in_gi.out_addr;
  param.output_addr = gi.out_addr;
  param.buffer_addr = gi.buffer_addr;
  param.input_n = sec_info.n_slice;
  param.input_c = c;
  param.input_h = sec_info.h_slice;
  param.input_w = sec_info.w_slice;
  param.scale_val = getMultiplier();
  param.rshift_num = getRshift();
  param.input_dtype = BM168x::getDataType(getInput());
  param.scale_dtype = param.scale_val < 0 ? DTYPE_INT8 : DTYPE_UINT8;
  param.output_dtype = BM168x::getDataType(getOutput());
  param.round_mode = ROUND_UP;
  BM168x::call_local_func("backend_api_mulshift_local", &param, sizeof(param));
}

// dynamic codegen
int64_t tpu::MulShiftOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_mulshift_local_param_t);
  dyn_mulshift_local_param_t param = {0};
  auto gi = getGroupInfo(0, 0);
  param.buffer_addr = gi.buffer_addr;
  param.common.scale_val = getMultiplier();
  param.common.rshift_num = getRshift();
  param.common.scale_dtype =
      param.common.scale_val < 0 ? DTYPE_INT8 : DTYPE_UINT8;
  param.common.output_dtype = BM168x::getDataType(getOutput());
  param.common.round_mode = ROUND_UP;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MulShiftOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }
