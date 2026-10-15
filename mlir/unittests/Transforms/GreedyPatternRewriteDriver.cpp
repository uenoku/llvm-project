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
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class GreedyPatternRewriteDriverTest : public ::testing::TestWithParam<bool> {
protected:
  GreedyPatternRewriteDriverTest() {
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<cf::ControlFlowDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.allowUnregisteredDialects();
    config.setUseTopDownTraversal(GetParam())
        .enableFolding(false)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  }

  void checkChanged(StringRef source) {
    FrozenRewritePatternSet patterns{RewritePatternSet(&context)};
    for (int64_t limit :
         {int64_t{0}, int64_t{1}, GreedyRewriteConfig::kNoLimit}) {
      SCOPED_TRACE(limit);
      auto module = parseSourceString<ModuleOp>(source, &context);
      ASSERT_TRUE(module);
      OperationFingerPrint before(*module);
      config.setMaxIterations(limit);
      bool changed = limit == 0;
      auto result = applyPatternsGreedily(*module, patterns, config, &changed);
      EXPECT_EQ(changed, limit != 0);
      EXPECT_EQ(succeeded(result), limit != 1);
      if (limit == 0) {
        EXPECT_EQ(before, OperationFingerPrint(*module));
        continue;
      }

      // Once simplified, another run is a no-op.
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
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
  checkChanged(R"mlir(
    func.func @f(%arg0: f32) {
      cf.br ^bb1(%arg0: f32)
    ^bb1(%unused: f32):
      return
    }
  )mlir");
}

TEST(OperationFolderTest, InsertKnownConstant) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.allowUnregisteredDialects();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    "test.side_effect"() : () -> ()
    %0 = arith.constant 42 : i32
    %1 = arith.constant 42 : i32
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  Operation *sideEffect = &module->getBody()->front();
  Operation *constant = sideEffect->getNextNode();
  Operation *duplicate = constant->getNextNode();

  using Result = OperationFolder::ConstantInsertionResult;
  OperationFolder folder(&context);
  EXPECT_EQ(folder.insertKnownConstant(constant), Result::Hoisted);
  EXPECT_EQ(folder.insertKnownConstant(constant), Result::Unchanged);
  EXPECT_EQ(folder.insertKnownConstant(duplicate), Result::Replaced);

  // Folder-owned constants can be hoisted again.
  sideEffect->moveBefore(constant);
  EXPECT_EQ(folder.insertKnownConstant(constant), Result::Hoisted);
}

INSTANTIATE_TEST_SUITE_P(TraversalOrder, GreedyPatternRewriteDriverTest,
                         ::testing::Bool());
} // namespace
