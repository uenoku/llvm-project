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
  dbgs() << "Prediction try"
         << "\n";
  if (!In)
    return true;
    // dbgs() << "Prediction Running"
    //        << "\n";
#ifndef SIMPLE_LOG
  if (In->PassName == "FunctionToLoopPassAdaptor<llvm::LICMPass>") {
    auto res = predict_LICM(In->Features);
    //    (res ? NumPredictLICMTrue : NumPredictLICMFalse)++;
    return res;
  }
  if (In->PassName == "FunctionToLoopPassAdaptor<llvm::LoopRotatePass>") {
    auto res = predict_LoopRotatePass(In->Features);
    //    (res ? NumPredictLoopRotatePassTrue :
    //    NumPredictLoopRotatePassFalse)++;
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
#else
  auto Models = Ctx.getPredictor();
  if (!Models) {
    Ctx.setPredictor(std::make_unique<FunctionPassResultPredictionModel>());
    Models = Ctx.getPredictor();
  }
  auto Model = Models->get(In->PassName.str(), Ctx);
  dbgs() << "Trying to fetch model" << In->PassName.str() << " " << (bool)Model
         << "\n";
  if (!Model)
    return true;
  {
    FunctionPropertiesInfo &FPI = In->Features;
    Model->setFeature(0, FPI.BasicBlockCount);
    Model->setFeature(1, FPI.BasicBlockWithMoreThanTwoPredecessors);
    Model->setFeature(2, FPI.BasicBlockWithMoreThanTwoSuccessors);
    Model->setFeature(3, FPI.BasicBlockWithSinglePredecessor);
    Model->setFeature(4, FPI.BasicBlockWithSingleSuccessor);
    Model->setFeature(5, FPI.BasicBlockWithTwoPredecessors);
    Model->setFeature(6, FPI.BasicBlockWithTwoSuccessors);
    Model->setFeature(7, FPI.BigBasicBlock);
    Model->setFeature(8, FPI.BlocksReachedFromConditionalInstruction);
    Model->setFeature(9, FPI.CastInstCount);
    Model->setFeature(10, FPI.DirectCallsToDefinedFunctions);
    Model->setFeature(11, FPI.FloatingConstantOccurrences);
    Model->setFeature(12, FPI.FloatingPointInstCount);
    Model->setFeature(13, FPI.InstructionCount);
    Model->setFeature(14, FPI.IntegerConstantOccurrences);
    Model->setFeature(15, FPI.IntegerInstCount);
    Model->setFeature(16, FPI.MaxLoopDepth);
    Model->setFeature(17, FPI.MediumBasicBlock);
    Model->setFeature(18, FPI.OpCodeCount[1]);
    Model->setFeature(19, FPI.OpCodeCount[10]);
    Model->setFeature(20, FPI.OpCodeCount[11]);
    Model->setFeature(21, FPI.OpCodeCount[12]);
    Model->setFeature(22, FPI.OpCodeCount[13]);
    Model->setFeature(23, FPI.OpCodeCount[14]);
    Model->setFeature(24, FPI.OpCodeCount[15]);
    Model->setFeature(25, FPI.OpCodeCount[16]);
    Model->setFeature(26, FPI.OpCodeCount[17]);
    Model->setFeature(27, FPI.OpCodeCount[18]);
    Model->setFeature(28, FPI.OpCodeCount[19]);
    Model->setFeature(29, FPI.OpCodeCount[2]);
    Model->setFeature(30, FPI.OpCodeCount[20]);
    Model->setFeature(31, FPI.OpCodeCount[21]);
    Model->setFeature(32, FPI.OpCodeCount[22]);
    Model->setFeature(33, FPI.OpCodeCount[23]);
    Model->setFeature(34, FPI.OpCodeCount[24]);
    Model->setFeature(35, FPI.OpCodeCount[25]);
    Model->setFeature(36, FPI.OpCodeCount[26]);
    Model->setFeature(37, FPI.OpCodeCount[27]);
    Model->setFeature(38, FPI.OpCodeCount[28]);
    Model->setFeature(39, FPI.OpCodeCount[29]);
    Model->setFeature(40, FPI.OpCodeCount[3]);
    Model->setFeature(41, FPI.OpCodeCount[30]);
    Model->setFeature(42, FPI.OpCodeCount[31]);
    Model->setFeature(43, FPI.OpCodeCount[32]);
    Model->setFeature(44, FPI.OpCodeCount[33]);
    Model->setFeature(45, FPI.OpCodeCount[34]);
    Model->setFeature(46, FPI.OpCodeCount[35]);
    Model->setFeature(47, FPI.OpCodeCount[36]);
    Model->setFeature(48, FPI.OpCodeCount[37]);
    Model->setFeature(49, FPI.OpCodeCount[38]);
    Model->setFeature(50, FPI.OpCodeCount[39]);
    Model->setFeature(51, FPI.OpCodeCount[4]);
    Model->setFeature(52, FPI.OpCodeCount[40]);
    Model->setFeature(53, FPI.OpCodeCount[41]);
    Model->setFeature(54, FPI.OpCodeCount[42]);
    Model->setFeature(55, FPI.OpCodeCount[43]);
    Model->setFeature(56, FPI.OpCodeCount[44]);
    Model->setFeature(57, FPI.OpCodeCount[45]);
    Model->setFeature(58, FPI.OpCodeCount[46]);
    Model->setFeature(59, FPI.OpCodeCount[47]);
    Model->setFeature(60, FPI.OpCodeCount[48]);
    Model->setFeature(61, FPI.OpCodeCount[49]);
    Model->setFeature(62, FPI.OpCodeCount[5]);
    Model->setFeature(63, FPI.OpCodeCount[50]);
    Model->setFeature(64, FPI.OpCodeCount[51]);
    Model->setFeature(65, FPI.OpCodeCount[52]);
    Model->setFeature(66, FPI.OpCodeCount[53]);
    Model->setFeature(67, FPI.OpCodeCount[54]);
    Model->setFeature(68, FPI.OpCodeCount[55]);
    Model->setFeature(69, FPI.OpCodeCount[56]);
    Model->setFeature(70, FPI.OpCodeCount[57]);
    Model->setFeature(71, FPI.OpCodeCount[58]);
    Model->setFeature(72, FPI.OpCodeCount[59]);
    Model->setFeature(73, FPI.OpCodeCount[6]);
    Model->setFeature(74, FPI.OpCodeCount[60]);
    Model->setFeature(75, FPI.OpCodeCount[61]);
    Model->setFeature(76, FPI.OpCodeCount[62]);
    Model->setFeature(77, FPI.OpCodeCount[63]);
    Model->setFeature(78, FPI.OpCodeCount[64]);
    Model->setFeature(79, FPI.OpCodeCount[65]);
    Model->setFeature(80, FPI.OpCodeCount[66]);
    Model->setFeature(81, FPI.OpCodeCount[7]);
    Model->setFeature(82, FPI.OpCodeCount[8]);
    Model->setFeature(83, FPI.OpCodeCount[9]);
    Model->setFeature(84, FPI.SmallBasicBlock);
    Model->setFeature(85, FPI.TopLevelLoopCount);
    Model->setFeature(86, FPI.Uses);
  }
  auto res = Model->run();
  dbgs() << "Prediction Result" << In->PassName.str() << " " << res << "\n";
  return res;
  // if (!Pred->PassNameToModel.count(In->PassName.str()))
  //   return true;
  // MLModelRunner *Model = Pred->PassNameToModel[In->PassName.str()];
  // Model->setFeature((size_t)PassResultPredictionFeatureIndex::BasicBlockCount,
  //                   In->Features.BasicBlockCount);
  // Model->setFeature((size_t)PassResultPredictionFeatureIndex::InstructionCount,
  //                   In->Features.InstructionCount);
  //  return Model->run();
#endif
  return true;
}

template <>
PredictorInput *
MLPassResultPredictor<Module, ModuleAnalysisManager>::createInput(
    Module &IR, ModuleAnalysisManager &FAM, StringRef PassName) {
  return nullptr;
}
static const std::string registered[] = {"SROA",
                                         "LoopUnrollPass",
                                         "LoopSimplifyPass",
                                         "BDCEPass",
                                         "EarlyCSEPass",
                                         "JumpThreadingPass",
                                         "TailCallElimPass",
                                         "SLPVectorizerPass",
                                         "LCSSAPass",
                                         "SimplifyCFGPass",
                                         "InstSimplifyPass", 
                                         "GVN"
                                         };
template <>
PredictorInput *
MLPassResultPredictor<Function, FunctionAnalysisManager>::createInput(
    Function &IR, FunctionAnalysisManager &FAM, StringRef PassName) {
  if (!RunPrediction)
    return nullptr;
  // dbgs() << "HOGE" << PassName << "\n";
#ifdef SIMPLE_LOG
  if (PassName == "FunctionToLoopPassAdaptor<llvm::LICMPass>" ||
      PassName == "FunctionToLoopPassAdaptor<llvm::LoopRotatePass>" ||
      PassName == "GVN" || PassName == "SROA" ||
      PassName == "ReassociatePass" || PassName == "InstSimplifyPass") {
    FunctionPropertiesAnalysis Ana;
    FunctionPropertiesAnalysis::Result Result = Ana.run(IR, FAM);
    return new PredictorInput{Result, FAM.PassResults[&IR], PassName};
  }
#else
  for (int i = 0; i < 12; i++)
    if (registered[i] == PassName) {
      FunctionPropertiesAnalysis Ana;
      FunctionPropertiesAnalysis::Result Result = Ana.run(IR, FAM);
      return new PredictorInput{Result, FAM.PassResults[&IR], PassName};
    }
#endif
  return nullptr;
}
} // namespace llvm
