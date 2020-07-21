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

static cl::opt<bool> StorePassResult("store-pass-result", cl::Hidden,
                                     cl::desc("store feature vector"),
                                     cl::init(false));

template <>
bool MLPassResultPredictor<Module, ModuleAnalysisManager>::predict(
    Module &IR, ModuleAnalysisManager &MAM, StringRef PassName,
    std::vector<std::pair<StringRef, bool>> PassResult
    ) {
  bool Result = true;
  dbgs() << "Run PasResultPredictor for " << PassName << "\n";
  return Result;
}

template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::predict(
    Function &IR, FunctionAnalysisManager &FAM, StringRef PassName,  
    std::vector<std::pair<StringRef, bool>> PassResult
    ) {

  bool Result = true;
  if (StorePassResult) {
    InlineFeaturesAnalysis Ana;
    InlineFeaturesAnalysis::Result Result = Ana.run(IR, FAM);
    json::Value json = Result.toJSON();
    json::Array json_array;
    for(auto [pass, result]: PassResult){
      json::Object object;
      object.try_emplace(pass, result);
      json_array.push_back(std::move(object));
    }
    json::Object obj;
    obj.try_emplace("feature", json);
    obj.try_emplace("pass", std::move(json_array));
    json::Value b(std::move(obj));
    dbgs () << b << "\n";
  }
  dbgs() << "Run PasResultPredictor for " << PassName << "\n";
  return Result;
}
