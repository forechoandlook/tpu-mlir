//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void SubLowering::LoweringINT8(PatternRewriter &rewriter, top::SubOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  std::vector<double> coeff_v(nInputs, 1.0);
  int64_t o_zp;
  double o_scale;
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);

  if (op.getCoeff().has_value()) {
    int idx = 0;
    for (auto v : op.getCoeff().value()) {
      coeff_v[idx++] = v.cast<FloatAttr>().getValueAsDouble();
    }
  }

  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    int64_t in_zp;
    double in_scale;
    module::getScaleAndZeroPoint(input, in_scale, in_zp, asymmetric);
    rshift_v[i] =
        calRightShiftNumUseCblas(coeff_v[i], in_scale, o_scale, BITS_INT8);
    float scale = 1.0 * (1 << rshift_v[i]) * in_scale / o_scale;
    int8_t multiplier_int8 = 0;
    float coeff = coeff_v[i];
    quantizeToInt8(&coeff, &multiplier_int8, 1, scale);
    multiplier_v[i] = (double)multiplier_int8;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::SubOp>(op, newType, operands, attrs);
}

void SubLowering::LoweringF32(PatternRewriter &rewriter, top::SubOp op) const {
  lowering_common_f32<tpu::SubOp>(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
