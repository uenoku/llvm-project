//===- RegionUtils.cpp - Region utility tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(RegionUtilsTest, RegionDCEReportsDeadBlockArgumentRemoval) {
  MLIRContext context;
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%arg0: f32) -> f32 {
      cf.br ^bb1(%arg0, %arg0, %arg0: f32, f32, f32)
    ^bb1(%dead0: f32, %live: f32, %dead1: f32):
      return %live: f32
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = cast<func::FuncOp>(module->getBody()->front());
  Region &body = func.getBody();
  Block &entry = body.front();
  Block &successor = body.back();
  auto branch = cast<cf::BranchOp>(entry.getTerminator());
  auto ret = cast<func::ReturnOp>(successor.getTerminator());

  // Removing only arguments, without erasing any operations, is still a change.
  IRRewriter rewriter(&context);
  ASSERT_TRUE(succeeded(runRegionDCE(rewriter, body)));
  EXPECT_EQ(body.getBlocks().size(), 2u);
  EXPECT_EQ(entry.getOperations().size(), 1u);
  EXPECT_EQ(successor.getOperations().size(), 1u);
  EXPECT_EQ(entry.getNumArguments(), 1u);
  ASSERT_EQ(successor.getNumArguments(), 1u);
  ASSERT_EQ(branch.getDestOperands().size(), 1u);
  EXPECT_EQ(branch.getDestOperands()[0], entry.getArgument(0));
  EXPECT_EQ(ret.getOperand(0), successor.getArgument(0));
  EXPECT_TRUE(succeeded(verify(*module)));

  // The remaining argument is live, so another run must report no change.
  EXPECT_TRUE(failed(runRegionDCE(rewriter, body)));
}
} // namespace
