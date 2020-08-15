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
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/FunctionPassResultPredictionModel.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/InlineCost.h"
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
static cl::opt<bool> RunPrediction("run-prediction", cl::Hidden,
                                   cl::desc("pass result predictor"),
                                   cl::init(false));
STATISTIC(NumPredictSROAFalse, "Number of prediction of false for SROA");
STATISTIC(NumPredictSROATrue, "Number of prediction of true for SROA");
STATISTIC(NumPredictGVNFalse, "Number of prediction of false for GVN");
STATISTIC(NumPredictGVNTrue, "Number of prediction of true for GVN");
STATISTIC(NumPredictLICMFalse, "Number of prediction of false for LICM");
STATISTIC(NumPredictLICMTrue, "Number of prediction of true for LICM");
STATISTIC(NumPredictInstSimplifyPassFalse,
          "Number of prediction of false for InstSimplify");
STATISTIC(NumPredictInstSimplifyPassTrue,
          "Number of prediction of true for InstSimplify");
STATISTIC(NumPredictReassociatePassFalse,
          "Number of prediction of false for Reassociate");
STATISTIC(NumPredictReassociatePassTrue,
          "Number of prediction of true for Reassociate");
STATISTIC(NumPredictLoopRotatePassFalse,
          "Number of prediction of false for LoopRotate");
STATISTIC(NumPredictLoopRotatePassTrue,
          "Number of prediction of true for LoopRotate");

struct PredictorInput {
  FunctionPropertiesAnalysis::Result Features;
  std::vector<std::pair<StringRef, bool>> PassResults;
  StringRef PassName;
  json::Value toJSON() const {
    json::Value json = Features.toJSON();
    json::Object obj;
    // json::Array json_array;
    // for (auto [pass, result] : PassResults) {
    //   json::Object object;
    //   object.try_emplace(pass, result);
    //   json_array.push_back(std::move(object));
    // }
    // obj.try_emplace("pass_history", std::move(json_array));
    obj.try_emplace("feature", json);
    obj.try_emplace("pass", PassName);
    return obj;
  }
};
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dump(
    StringRef s, PredictorInput *in, bool modified, raw_ostream &OS) {
  return;
}
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dump(
    StringRef s, PredictorInput *in, bool modified, raw_ostream &OS) {
  json::Object obj;
  if (!in || !StorePassResult)
    return;
  obj["IR_name"] = s;
  obj["input"] = in->toJSON();
  obj["modified"] = modified;
  json::Value v = std::move(obj);

  OS << v << "\n";
}
template <>
bool MLPassResultPredictor<Module, ModuleAnalysisManager>::predict(
    PredictorInput *In, LLVMContext &Ctx) {
  return true;
}
#include "./Prediction/predictor_FunctionToLoopPassAdaptor<llvm::LICMPass>.inc"
#include "./Prediction/predictor_FunctionToLoopPassAdaptor<llvm::LoopRotatePass>.inc"
#include "./Prediction/predictor_GVN.inc"
#include "./Prediction/predictor_InstSimplifyPass.inc"
#include "./Prediction/predictor_LoopUnrollPass.inc"
#include "./Prediction/predictor_ReassociatePass.inc"
#include "./Prediction/predictor_SROA.inc"

template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::predict(
    PredictorInput *In, LLVMContext &Ctx) {
  // dbgs() << "Prediction try"
  //        << "\n";
  if (!In)
    return true;
  // dbgs() << "Prediction Running"
  //        << "\n";
  if (In->PassName == "FunctionToLoopPassAdaptor<llvm::LICMPass>") {
    auto res = predict_LICM(In->Features);
//    (res ? NumPredictLICMTrue : NumPredictLICMFalse)++;
    return res;
  }
  if (In->PassName == "FunctionToLoopPassAdaptor<llvm::LoopRotatePass>") {
    auto res = predict_LoopRotatePass(In->Features);
//    (res ? NumPredictLoopRotatePassTrue : NumPredictLoopRotatePassFalse)++;
    return res;
  }
  if (In->PassName == "GVN") {
    auto res = predict_GVN(In->Features);
//    (res ? NumPredictGVNTrue : NumPredictGVNFalse)++;
    return res;
  }
  if (In->PassName == "ReassociatePass")
    return predict_ReassociatePass(In->Features);
  if (In->PassName == "SROA")
    return predict_ReassociatePass(In->Features);
  if (In->PassName == "InstSimplifyPass")
    return predict_InstSimplifyPass(In->Features);
  return true;
  // FunctionPassResultPredictionModel *Pred = Ctx.getPredictor();
  // if (!Pred->PassNameToModel.count(In->PassName.str()))
  //   return true;
  // MLModelRunner *Model = Pred->PassNameToModel[In->PassName.str()];
  // Model->setFeature((size_t)PassResultPredictionFeatureIndex::BasicBlockCount,
  //                   In->Features.BasicBlockCount);
  // Model->setFeature((size_t)PassResultPredictionFeatureIndex::InstructionCount,
  //                   In->Features.InstructionCount);
  //  return Model->run();
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
  if (!RunPrediction)
    return nullptr;
  // dbgs() << "HOGE" << PassName << "\n";
  if (PassName == "FunctionToLoopPassAdaptor<llvm::LICMPass>" ||
      PassName == "FunctionToLoopPassAdaptor<llvm::LoopRotatePass>" ||
      PassName == "GVN" || PassName == "SROA" ||
      PassName == "ReassociatePass" || PassName == "InstSimplifyPass") {
    FunctionPropertiesAnalysis Ana;
    FunctionPropertiesAnalysis::Result Result = Ana.run(IR, FAM);
    return new PredictorInput{Result, FAM.PassResults[&IR], PassName};
  }
  return nullptr;
}
} // namespace llvm
