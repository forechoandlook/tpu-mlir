//===- Bufferization.h - Bufferization dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATION_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATION_H_

#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// Bufferization Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizationOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Bufferization Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Bufferization/IR/BufferizationOps.h.inc"

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bufferization {
/// Try to cast the given ranked MemRef-typed value to the given ranked MemRef
/// type. Insert a reallocation + copy if it cannot be statically guaranteed
/// that a direct cast would be valid.
///
/// E.g., when casting from a ranked MemRef type with dynamic layout to a ranked
/// MemRef type with static layout, it is not statically known whether the cast
/// will succeed or not. Such `memref.cast` ops may fail at runtime. This
/// function never generates such casts and conservatively inserts a copy.
///
/// This function returns `failure()` in case of unsupported casts. E.g., casts
/// with differing element types or memory spaces.
FailureOr<Value> castOrReallocMemRefValue(OpBuilder &b, Value value,
                                          MemRefType type);

/// Try to fold to_memref(to_tensor(x)). If x's type and the result type of the
/// to_memref op are different, a memref.cast is needed.
LogicalResult foldToMemrefToTensorPair(RewriterBase &rewriter,
                                       ToMemrefOp toMemref,
                                       bool allowSameType = true);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATION_H_
