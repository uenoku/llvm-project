//===- GreedyPatternRewriteDriver.cpp - Tests ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class GreedyPatternRewriteDriverTest : public ::testing::TestWithParam<bool> {
protected:
  GreedyPatternRewriteDriverTest() {
    context.getOrLoadDialect<arith::ArithDialect>();
    context.allowUnregisteredDialects();
    config.setUseTopDownTraversal(GetParam())
        .enableFolding(false)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  }

  void checkChanged(StringRef source) {
    // Also check that changes in the last allowed iteration are reported, even
    // though the rewrite does not converge.
    for (int64_t maxIterations : {int64_t{1}, GreedyRewriteConfig::kNoLimit}) {
      auto module = parseSourceString<ModuleOp>(source, &context);
      ASSERT_TRUE(module);
      config.setMaxIterations(maxIterations);
      FrozenRewritePatternSet patterns{RewritePatternSet(&context)};
      bool changed = false;
      LogicalResult result =
          applyPatternsGreedily(*module, patterns, config, &changed);
      EXPECT_TRUE(changed);
      EXPECT_EQ(succeeded(result),
                maxIterations == GreedyRewriteConfig::kNoLimit);

      // Once the IR is simplified, another run is a no-op.
      changed = true;
      EXPECT_TRUE(succeeded(
          applyPatternsGreedily(*module, patterns, config, &changed)));
      EXPECT_FALSE(changed);
    }
  }

  MLIRContext context;
  GreedyRewriteConfig config;
};

TEST_P(GreedyPatternRewriteDriverTest, ConstantCSE) {
  checkChanged(R"mlir(
    %0 = arith.constant 42 : i32
    %1 = arith.constant 42 : i32
    "test.use"(%0, %1) : (i32, i32) -> ()
  )mlir");
}

TEST_P(GreedyPatternRewriteDriverTest, ConstantHoisting) {
  checkChanged(R"mlir(
    "test.side_effect"() : () -> ()
    %0 = arith.constant 42 : i32
    "test.use"(%0) : (i32) -> ()
  )mlir");
}

TEST_P(GreedyPatternRewriteDriverTest, CSEBetweenIterations) {
  config.enableConstantCSE(false).enableCSEBetweenIterations(true);
  checkChanged(R"mlir(
    %0 = "test.source"() : () -> i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %0, %0 : i32
    "test.use"(%1, %2) : (i32, i32) -> ()
  )mlir");
}

TEST_P(GreedyPatternRewriteDriverTest, DeadBlockArguments) {
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
  checkChanged(R"mlir(
    func.func @f(%arg0: f32) {
      cf.br ^bb1(%arg0: f32)
    ^bb1(%unused: f32):
      return
    }
  )mlir");
}

TEST_P(GreedyPatternRewriteDriverTest, NoIterations) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    %0 = arith.constant 42 : i32
    %1 = arith.constant 42 : i32
    "test.use"(%0, %1) : (i32, i32) -> ()
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  FrozenRewritePatternSet patterns{RewritePatternSet(&context)};
  config.setMaxIterations(0);
  bool changed = true;
  EXPECT_TRUE(
      succeeded(applyPatternsGreedily(*module, patterns, config, &changed)));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->getBody()->getOperations().size(), 3u);
}

TEST(OperationFolderTest, InsertKnownConstantAccumulatesChanges) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.allowUnregisteredDialects();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    %0 = arith.constant 1 : i32
    %1 = arith.constant 2 : i32
    %2 = arith.constant 1 : i32
    "test.use"(%0, %1, %2) : (i32, i32, i32) -> ()
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  SmallVector<arith::ConstantOp> constants;
  module->walk([&](arith::ConstantOp op) { constants.push_back(op); });
  ASSERT_EQ(constants.size(), 3u);

  OperationFolder folder(&context);
  bool changed = false;
  EXPECT_TRUE(folder.insertKnownConstant(constants[0], {}, &changed));
  EXPECT_FALSE(changed);

  // Registering a constant without moving it must not clear an earlier change.
  changed = true;
  EXPECT_TRUE(folder.insertKnownConstant(constants[1], {}, &changed));
  EXPECT_TRUE(changed);

  changed = false;
  EXPECT_FALSE(folder.insertKnownConstant(constants[2], {}, &changed));
  EXPECT_TRUE(changed);

  // Re-registering a folder-owned constant must preserve the CSE change too.
  EXPECT_TRUE(folder.insertKnownConstant(constants[0], {}, &changed));
  EXPECT_TRUE(changed);
}

INSTANTIATE_TEST_SUITE_P(TraversalOrder, GreedyPatternRewriteDriverTest,
                         ::testing::Bool());
} // namespace
