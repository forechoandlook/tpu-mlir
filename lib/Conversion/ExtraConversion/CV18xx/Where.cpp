//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"

namespace tpu_mlir {
namespace cv18xx {

LogicalResult ConvertWhereOp::matchAndRewrite(top::WhereOp op,
                                              PatternRewriter &rewriter) const {
  //out = input[0] * input[1] + (1 - input[0]) * input[2]
  Value input0 = op.getOperand(0);
  Value input1 = op.getOperand(1);
  Value input2 = op.getOperand(2);
  Value ori_out = op.getOutput();
  std::string name = module::getName(ori_out).str();
  std::vector<int64_t> output_shape = module::getShape(ori_out);
  std::vector<int64_t> input0_shape = module::getShape(input0);
  std::vector<int64_t> input1_shape = module::getShape(input1);
  std::vector<int64_t> input2_shape = module::getShape(input2);
  int64_t num_input0 = module::getNumElements(input0);
  int64_t num_input1 = module::getNumElements(input1);
  int64_t num_input2 = module::getNumElements(input2);
  //cv18xx only support one operand broadcast now.
  assert((input0_shape == output_shape || input1_shape == output_shape || input2_shape == output_shape));
  bool isCali = false;
  double out_thr, in1_thr, in2_thr;
  if (module::isCalibratedType(ori_out)) {
    isCali = true;
    auto otype = module::getCalibratedType(ori_out);
    auto in1_type = module::getCalibratedType(input1);
    auto in2_type = module::getCalibratedType(input2);
    out_thr = otype.getMax();
    in1_thr = in1_type.getMax();
    in2_thr = in2_type.getMax();
  }
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  rewriter.setInsertionPointAfterValue(ori_out);

  //create input[0] * input[1]
  operands.emplace_back(input0);
  operands.emplace_back(input1);
  std::vector<int64_t> out1_shape = (num_input0 > num_input1) ? input0_shape : input1_shape;
  auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
  RankedTensorType type1;
  if (isCali) {
    auto caliType = quant::CalibratedQuantizedType::get(rewriter.getF32Type(),
                    -in1_thr, in1_thr);
    type1 = RankedTensorType::get(out1_shape, caliType);
  } else {
    type1 = RankedTensorType::get(out1_shape, rewriter.getF32Type());
  }
  auto mulOp1 = rewriter.create<top::MulOp>(loc1, type1, operands, attrs);
  auto out1 = mulOp1.getOutput();


  //create input[0] * input[2]
  operands.clear();
  attrs.clear();
  operands.emplace_back(input0);
  operands.emplace_back(input2);
  rewriter.setInsertionPointAfterValue(out1);
  std::vector<int64_t> out2_shape = (num_input0 > num_input2) ? input0_shape : input2_shape;
  auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "mul2"));
  RankedTensorType type2;
  if (isCali) {
    auto caliType = quant::CalibratedQuantizedType::get(rewriter.getF32Type(),
                    -in2_thr, in2_thr);
    type2 = RankedTensorType::get(out2_shape, caliType);
  } else {
    type2 = RankedTensorType::get(out2_shape, rewriter.getF32Type());
  }
  auto mulOp2 = rewriter.create<top::MulOp>(loc2, type2, operands, attrs);
  auto out2 = mulOp2.getOutput();

  //create input[2] - input[0] * input[2]
  attrs.clear();
  operands.clear();
  operands.emplace_back(input2);
  operands.emplace_back(out2);
  rewriter.setInsertionPointAfterValue(out2);
  auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "_sub1"));
  auto subOp1 = rewriter.create<top::SubOp>(loc3, type2, operands, attrs);
  auto out3 = subOp1.getOutput();

  //create (input[0] * input[1]) + (input[2] - input[0] * input[2])
  attrs.clear();
  operands.clear();
  operands.emplace_back(out1);
  operands.emplace_back(out3);
  rewriter.setInsertionPointAfterValue(out3);
  auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
  RankedTensorType type4;
  if (isCali) {
    auto caliType = quant::CalibratedQuantizedType::get(rewriter.getF32Type(),
                    -out_thr, out_thr);
    type4 = RankedTensorType::get(output_shape, caliType);
  } else {
    type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
  }
  auto add2Op = rewriter.create<top::AddOp>(loc4, type4, operands, attrs);
  auto out4 = add2Op.getOutput();
  rewriter.replaceAllUsesWith(ori_out, out4);
  rewriter.eraseOp(op);

  return success();
}

} // namespace cv18xx
} // namespace tpu_mlir
