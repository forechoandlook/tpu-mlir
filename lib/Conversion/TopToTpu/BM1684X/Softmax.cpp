//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void SoftmaxLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.input());
  auto none = Module::getNoneOp(op);
  for (int i = 0; i < 4; i++) {
    operands.push_back(none);
  }
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(op, op.output().getType(),
                                              operands, op->getAttrs());
}

void SoftmaxLowering::LoweringINT8(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SoftmaxOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SoftmaxOp op) const {
  if (Quant::isUniformQuantized(op.input(), op.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  int64_t zeropoint;
  double i_scale;
  Quant::getScaleAndZeroPoint(op.input(), i_scale, zeropoint, true);
  std::vector<float> table(256, 0.0f);
  auto beta_v = op.beta().convertToDouble();
  auto scale = -i_scale * beta_v;

  // const int int_bits = 5;
  // const double_t multiplier_max = (1ULL << 31) - 1;
  // double_t multiplier_real =
  //     std::min(i_scale * beta_v * (1 << (31 - int_bits)), multiplier_max);
  // int32_t multi, shift;
  // QuantizeMultiplier(multiplier_real, &multi, &shift);
  // double max_input_rescaled =
  //     1.0 * ((1 << int_bits) - 1) * (1LL << (31 - int_bits)) / (1LL <<
  //     shift);
  // int32_t diff_min = -1 *
  // static_cast<int32_t>(std::floor(max_input_rescaled)); std::vector<int32_t>
  // table(256, 0); for (int i = 0; i < 256; ++i) {
  //   int32_t input_diff_rescaled = MultiplyByQuantizedMultiplier(-i, multi,
  //   shift); table[i] = exp_on_negative_values(input_diff_rescaled, int_bits);
  // }

  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(scale * i);
  }
  auto table_opd = create_lookup_table(op, table);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(
      op, op.output().getType(),
      ValueRange{op.input(), table_opd, Module::getNoneOp(op.getOperation()),
                 Module::getNoneOp(op.getOperation()),
                 Module::getNoneOp(op.getOperation())},
      attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
