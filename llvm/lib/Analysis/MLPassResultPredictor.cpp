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

#define DEBUG_TYPE "pass-result-predictor-ml"

namespace llvm {
static cl::opt<bool> StorePassResult("store-pass-result", cl::Hidden,
                                     cl::desc("store feature vector"),
                                     cl::init(false));

struct PredictorInput {
  InlineFeaturesAnalysis::Result Features;
  std::vector<std::pair<StringRef, bool>> PassResults;
  StringRef PassName;
  json::Value toJSON() const {
    json::Value json = Features.toJSON();
    json::Array json_array;
    for (auto [pass, result] : PassResults) {
      json::Object object;
      object.try_emplace(pass, result);
      json_array.push_back(std::move(object));
    }
    json::Object obj;
    obj.try_emplace("feature", json);
    obj.try_emplace("pass_history", std::move(json_array));
    obj.try_emplace("pass", PassName);
    return obj;
  }
};
template<>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dump(StringRef s, PredictorInput* in, bool modified, raw_ostream& OS){
  json::Object obj;
  if(!in)return;
  obj["IR_name"] = s;
  obj["input"] = in->toJSON();
  obj["modified"] = modified;
  json::Value v = std::move(obj);
  OS << v << "\n";
}
template<>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dump(StringRef s, PredictorInput* in, bool modified, raw_ostream& OS){
  json::Object obj;
  if(!in)return;
  obj["IR_name"] = s;
  obj["input"] = in->toJSON();
  obj["modified"] = modified;
  json::Value v = std::move(obj);
  OS << v << "\n";
}
template <>
bool MLPassResultPredictor<Module, ModuleAnalysisManager>::predict(
    PredictorInput *In) {
  return true;
}

template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::predict(
    PredictorInput *In) {
  return true;
}

template <>
PredictorInput *
MLPassResultPredictor<Module, ModuleAnalysisManager>::createInput(
    Module &IR, ModuleAnalysisManager &FAM, StringRef PassName) {
  return nullptr;
}

template <>
PredictorInput *
MLPassResultPredictor<Function, FunctionAnalysisManager>::createInput(
    Function &IR, FunctionAnalysisManager &FAM, StringRef PassName) {
  InlineFeaturesAnalysis Ana;
  InlineFeaturesAnalysis::Result Result = Ana.run(IR, FAM);
  return new PredictorInput{Result, FAM.PassResults[&IR], PassName};
}
} // namespace llvm
