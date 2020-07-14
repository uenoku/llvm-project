//===- MLPassResultPredictor.cpp - machine learned Pass result predictor
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface between the inliner and a learned model.
// It delegates model evaluation to either the AOT compiled model (the
// 'release' mode) or a runtime-loaded model (the 'development' case).
//
//===----------------------------------------------------------------------===//
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InlineFeaturesAnalysis.h"
#include "llvm/Analysis/MLInlineAdvisor.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/MLPassResultPredictor.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

using namespace llvm;

#define DEBUG_TYPE "pass-result-predictor-ml"

template <>
bool MLPassResultPredictor<Module>::predict(Module &IR, StringRef PassName) {
  bool Result = true;
  dbgs() << "Run PasResultPredictor for " << PassName << "\n";
  return Result;
}

template <>
bool MLPassResultPredictor<Function>::predict(Function &IR,
                                              StringRef PassName) {
  bool Result = true;
  dbgs() << "Run PasResultPredictor for " << PassName << "\n";
  return Result;
}
