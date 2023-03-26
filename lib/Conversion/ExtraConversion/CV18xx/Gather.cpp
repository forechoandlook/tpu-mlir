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

LogicalResult
ConvertGatherOp::matchAndRewrite(top::GatherOp op,
                                 PatternRewriter &rewriter) const {
  // for transform decode's index op
  Value input = op.getInput();
  Value indices = op.getIndices();
  Value ori_out = op.getOutput();
  std::string name = module::getName(ori_out).str();
  uint64_t axis = op.getAxis();
  std::vector<int64_t> input_shape = module::getShape(input);
  std::vector<int64_t> output_shape = module::getShape(ori_out);
  std::vector<int64_t> indices_shape = module::getShape(indices);
  bool need_convert = (axis == 1 && indices_shape.size() == 0 && input_shape.size() == 3 && input_shape[0] == 1
                          && !(isa<top::WeightOp>(input.getDefiningOp())));
  if (need_convert) {
    //conver to reshapeOp + new GatherOp
    rewriter.setInsertionPointAfterValue(ori_out);
    double in_thr, out_thr;
    RankedTensorType type1, type2;
    if (module::isCalibratedType(ori_out)) {
      auto itype = module::getCalibratedType(input);
      auto otype = module::getCalibratedType(ori_out);
      auto caliType1 = quant::CalibratedQuantizedType::get(rewriter.getF32Type(),
                        -in_thr, in_thr);
      auto caliType2 = quant::CalibratedQuantizedType::get(rewriter.getF32Type(),
                        -out_thr, out_thr);
      type1 = RankedTensorType::get({input_shape[1], input_shape[2]}, caliType1);
      type2 = RankedTensorType::get(output_shape, caliType2);
    } else {
      type1 = RankedTensorType::get({input_shape[1], input_shape[2]}, rewriter.getF32Type());
      type2 = ori_out.getType().cast<RankedTensorType>();
    }
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    operands.emplace_back(input);
    auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_reshape"));
    auto reshapeOp = rewriter.create<top::ReshapeOp>(loc1, type1, operands, attrs);
    auto out1 = reshapeOp.getOutput();
    operands.clear();
    operands.emplace_back(out1);
    operands.emplace_back(indices);
    attrs.emplace_back(rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(0)));
    auto loc2 = NameLoc::get(rewriter.getStringAttr(name));
    auto newOp = rewriter.create<top::GatherOp>(loc2, type2, operands, attrs);
    auto newOut = newOp.getOutput();
    rewriter.replaceAllUsesWith(ori_out, newOut);
    rewriter.eraseOp(op);
  } else {
    return failure();
  }
  return success();
}
} // namespace cv18xx
} // namespace tpu_mlir
