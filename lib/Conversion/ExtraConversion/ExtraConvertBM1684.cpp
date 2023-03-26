//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertBM1684.h"
#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertBM1684X.h"
namespace tpu_mlir {

namespace bm1684 {

void populateDoExtraConversionPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      tpu_mlir::bm1684x::ConvertMatMulWithRightTranspose
  >(patterns->getContext());
  // clang-format on
}

} // namespace bm1684
} // namespace tpu_mlir
