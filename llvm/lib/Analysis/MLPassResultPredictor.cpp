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
#include <sys/types.h>
#include <unistd.h>
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
#include "llvm/Analysis/PassResultAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/CrossDSOCFI.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/GlobalSplit.h"
#include "llvm/Transforms/IPO/HotColdSplitting.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/Transforms/IPO/MergeFunctions.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/IPO/PartialInlining.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/IPO/SampleProfile.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/SyntheticCountsPropagation.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/DivRemPairs.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GuardWidening.h"
#include "llvm/Transforms/Scalar/IVUsersPrinter.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InductiveRangeCheckElimination.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopAccessAnalysisPrinter.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopFuse.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopLoadElimination.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopPredication.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopSink.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopUnrollAndJamPass.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/LowerAtomic.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerGuardIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerWidenableCondition.h"
#include "llvm/Transforms/Scalar/MakeGuardsExplicit.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/SpeculateAroundPHIs.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Scalar/WarnMissedTransforms.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "pass-result-predictor-ml"

namespace llvm {
static llvm::cl::opt<unsigned> PredictionReuse("prediction-reuse", cl::Hidden,
                                               cl::desc("reuse-prediction."),
                                               cl::init(4));
static cl::opt<bool> StorePassResult("store-pass-result", cl::Hidden,
                                     cl::desc("store feature vector"),
                                     cl::init(false));
static cl::opt<bool> RunPrediction("run-prediction", cl::Hidden,
                                   cl::desc("pass result predictor"),
                                   cl::init(false));
static cl::opt<bool> ShowOrderPrediction("show-order", cl::Hidden,
                                         cl::desc("show config of input"),
                                         cl::init(false));
static cl::opt<bool> DumpAllResult("dump-all-result", cl::Hidden,
                                   cl::desc("show config of input"),
                                   cl::init(false));
static cl::opt<bool> NotRunForTrivial("not-run-for-trivial", cl::Hidden,
                                      cl::desc("show config of input"),
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

STATISTIC(NumPredictionForTrivial, "Number of prediction for prediction");
STATISTIC(NumPrediction, "Number of prediction");
STATISTIC(NumPredictionTrue, "Number of prediction true");
STATISTIC(NumPredictionFalse, "Number of prediction false");
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dumpAllResult(
    Function &IR, FunctionAnalysisManager &FAM) {
  if (!DumpAllResult)
    return;
  using Pass = detail::PassConcept<Function, FunctionAnalysisManager>;
  using PassModelT =
      detail::PassModel<Function, Pass, PreservedAnalyses, FunctionPassManager>;

  std::vector<std::shared_ptr<Pass>> to_run;
  FunctionPassManager FPM;
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(SROA());
  FPM.addPass(EarlyCSEPass());
  FPM.addPass(InstSimplifyPass());
  FPM.addPass(GVN());
  FPM.addPass(LoopUnrollPass());
  FPM.addPass(BDCEPass());
  FPM.addPass(TailCallElimPass());
  FPM.addPass(JumpThreadingPass());
  FPM.addPass(CorrelatedValuePropagationPass());
  FPM.addPass(InstCombinePass());

  std::vector<int> results;

  for (int i = 0; i < FPM.Passes.size(); i++) {

    ValueToValueMapTy VMap;
    // auto M = CloneModule(*IR.getParent());
    // M->materializeAll();
    // Function&F = *M->getFunction(IR.getName());
    Function &F = *CloneFunction(&IR, VMap);
    FAM.clear();
    PreservedAnalyses Res = FPM.Passes[i]->run(F, FAM);
    results.push_back(!Res.areAllPreserved());
    //   for(auto& B: F.getBasicBlockList()){
    //       for(auto &I:B){
    //         I.replaceAllUsesWith(UndefValue::get(I.getType()));
    //       }
    //  }
    VMap.clear();
    F.eraseFromParent();
  }
  LLVM_DEBUG(dbgs() << IR.getName() << " ";);
  json::Object res;
  json::Array res_arr;
  for (int i = 0; i < results.size(); i++) {
    // LLVM_DEBUG(dbgs() << (int)results[i];);
    LLVM_DEBUG(dbgs() << FPM.Passes[i]->name() << ",";);
    res_arr.push_back(bool(results[i]));
  }
  LLVM_DEBUG(dbgs() << "\n";);
  FunctionPropertiesAnalysis::Result feat =
      FAM.getResult<FunctionPropertiesAnalysis>(IR);
  json::Object obj;
  auto tt = feat.toVec();
  obj["feature"] = tt;
  obj["pass"] = std::move(res_arr);
  obj["name"] = IR.getName();
  json::Value v = std::move(obj);
  {
    int t = getpid();
    auto fname = "pass_all_data_" + std::to_string(t) + ".txt";
    std::error_code EC;
    auto os = std::make_unique<llvm::raw_fd_ostream>(fname, EC,
                                                     llvm::sys::fs::OF_Append);
    // *os << v << "\n";
    *os << IR.getName() << ",";
    for (auto s : tt) {
      *os << s << ",";
    }
    for (auto s : results) {
      *os << s << ",";
    }
    *os << "\n";
  }

  LLVM_DEBUG(dbgs() << v << "\n";);
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dumpAllResult(
    Module &IR, ModuleAnalysisManager &MAM) {}

#define SIMPLE_LOG
#define USE_NORMAL
// #define USE_NORMAL

#ifdef SIMPLE_LOG
using FuncPropType = FunctionPropertiesAnalysis;
#endif
#ifdef USE_NORMAL
using FuncPropType = FunctionPropertiesAnalysis;
#endif
#ifdef USE_MIN
using FuncPropType = FunctionPropertiesSmallAnalysis;
#endif
struct PredictorInput {
  FuncPropType::Result Features;
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
bool MLPassResultPredictor<Module, ModuleAnalysisManager>::predict_all(
    Module &In, ModuleAnalysisManager &MAM, StringRef name) {
  return true;
}
std::map<std::pair<Function*, int>, int> called;
template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::predict_all(
    Function &F, FunctionAnalysisManager &FAM, StringRef name) {
  if (!RunPrediction)
    return true;
  if (NotRunForTrivial && F.getInstructionCount() < 5) {
    NumPredictionForTrivial++;
    return true;
  }

  static bool first = true;
  static FunctionPassManager FPM;
  if (first) {
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(SROA());
    FPM.addPass(EarlyCSEPass());
    FPM.addPass(InstSimplifyPass());
    FPM.addPass(GVN());
    FPM.addPass(LoopUnrollPass());
    FPM.addPass(BDCEPass());
    FPM.addPass(TailCallElimPass());
    FPM.addPass(JumpThreadingPass());
    FPM.addPass(CorrelatedValuePropagationPass());
    FPM.addPass(InstCombinePass());
    first = false;
  }
  for (int i = 0; i < FPM.Passes.size(); i++) {
    if (FPM.Passes[i]->name() == name) {
      bool res = true;
      {
        FunctionPassResultAnalysis::Result *FPP =
            FAM.getCachedResult<FunctionPassResultAnalysis>(F);
        if (FPP) {
          LLVM_DEBUG(dbgs() << FPP->counter << " counter\n";);
          FPP->counter++;
          if (FPP->counter < PredictionReuse) {
            res = FPP->result[i];
          } else {
            res = FAM.getResult<FunctionPassResultAnalysis>(F).result[i];
          }
        } else {
          res = FAM.getResult<FunctionPassResultAnalysis>(F).result[i];
        }
      }
      NumPrediction++;
      (res ? NumPredictionTrue : NumPredictionFalse)++;
      return res;
    }
  }
  return true;
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
  LLVM_DEBUG(dbgs() << "Prediction try"
                    << "\n");
  if (!In)
    return true;

  bool ActuallyRun = true;
  auto calc = [&]() -> bool {
  // dbgs() << "Prediction Running"
  //        << "\n";
#ifdef SIMPLE_LOG
    //  std::cerr << "RUN SIMPLE LOG" << std::endl;
    if (In->PassName == "FunctionToLoopPassAdaptor<llvm::LICMPass>") {
      auto res = predict_LICM(In->Features);
      //    (res ? NumPredictLICMTrue : NumPredictLICMFalse)++;
      return res;
    }
    // if (In->PassName == "FunctionToLoopPassAdaptor<llvm::LoopRotatePass>") {
    //   auto res = predict_LoopRotatePass(In->Features);
    //   //    (res ? NumPredictLoopRotatePassTrue :
    //   //    NumPredictLoopRotatePassFalse)++;
    //   return res;
    // }
    if (In->PassName == "GVN") {
      return predict_GVN(In->Features);
    }

    if (In->PassName == "ReassociatePass")
      return predict_ReassociatePass(In->Features);
    if (In->PassName == "SROA")
      return predict_ReassociatePass(In->Features);
    if (In->PassName == "InstSimplifyPass")
      return predict_InstSimplifyPass(In->Features);
    ActuallyRun = false;
    return true;
#else
    auto Models = Ctx.getPredictor();
    if (!Models) {
      Ctx.setPredictor(std::make_unique<FunctionPassResultPredictionModel>());
      Models = Ctx.getPredictor();
    }
    auto Model = Models->get(In->PassName.str(), Ctx);
    LLVM_DEBUG(dbgs() << "Trying to fetch model" << In->PassName.str() << " "
                      << (bool)Model << "\n");
    if (!Model) {
      ActuallyRun = false;
      return true;
    }
    {
      auto &FPI = In->Features;
#endif

#ifdef USE_NORMAL
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
#else
      Model->setFeature(0, FPI.BasicBlockCount);
      Model->setFeature(1, FPI.IntegerConstantOccurrences);
      Model->setFeature(2, FPI.BasicBlockWithSingleSuccessor);
      Model->setFeature(3, FPI.GEP);
      Model->setFeature(4, FPI.MaxLoopDepth);
      Model->setFeature(5, FPI.Call);
      Model->setFeature(6, FPI.Alloca);
      Model->setFeature(7, FPI.Store);
      Model->setFeature(8, FPI.TopLevelLoopCount);
      Model->setFeature(9, FPI.IntegerInstCount);
      Model->setFeature(10, FPI.PHI);
      Model->setFeature(11, FPI.BasicBlockWithSinglePredecessor);
      Model->setFeature(12, FPI.Load);
      Model->setFeature(13, FPI.InstructionCount);
#endif
  } auto res = Model->run();
  LLVM_DEBUG(dbgs() << "Prediction Result" << In->PassName.str() << " " << res
                    << "\n");
  return res;
  // if (!Pred->PassNameToModel.count(In->PassName.str()))
  //   return true;
  // MLModelRunner *Model = Pred->PassNameToModel[In->PassName.str()];
  // Model->setFeature((size_t)PassResultPredictionFeatureIndex::BasicBlockCount,
  //                   In->Features.BasicBlockCount);
  // Model->setFeature((size_t)PassResultPredictionFeatureIndex::InstructionCount,
  //                   In->Features.InstructionCount);
  //  return Model->run();
};
bool res = calc();
if (ActuallyRun) {
  NumPrediction++;
  (res ? NumPredictionTrue : NumPredictionFalse)++;
}
return res;
} // namespace llvm

template <>
PredictorInput *
MLPassResultPredictor<Module, ModuleAnalysisManager>::createInput(
    Module &IR, ModuleAnalysisManager &FAM, StringRef PassName) {
  return nullptr;
}
static const std::vector<std::string> registered = {
    "SROA", "LoopUnrollPass",
    //                                         "LoopSimplifyPass",
    "BDCEPass", "EarlyCSEPass", "JumpThreadingPass", "TailCallElimPass",
    "SLPVectorizerPass", "LCSSAPass", "SimplifyCFGPass", "InstSimplifyPass",
    "GVN"};
template <>
PredictorInput *
MLPassResultPredictor<Function, FunctionAnalysisManager>::createInput(
    Function &IR, FunctionAnalysisManager &FAM, StringRef PassName) {

  LLVM_DEBUG(dbgs() << "Trying to create input for prediction" << PassName
                    << "\n");
  if (!RunPrediction)
    return nullptr;
  LLVM_DEBUG(dbgs() << "Prediction Start" << PassName << "\n");
#ifdef SIMPLE_LOG

  LLVM_DEBUG(dbgs() << "Simple logistic regression will be used" << PassName
                    << "\n");
  if (PassName == "FunctionToLoopPassAdaptor<llvm::LICMPass>" ||
      //     PassName == "FunctionToLoopPassAdaptor<llvm::LoopRotatePass>" ||
      PassName == "GVN" || PassName == "SROA" ||
      PassName == "ReassociatePass" || PassName == "InstSimplifyPass") {
    FunctionPropertiesAnalysis Ana;
    FunctionPropertiesAnalysis::Result Result = Ana.run(IR, FAM);
    return new PredictorInput{Result, FAM.PassResults[&IR], PassName};
  }
#else

  LLVM_DEBUG(dbgs() << "ML Model will be used" << PassName << "\n");
  for (int i = 0; i < registered.size(); i++)
    if (registered[i] == PassName) {
      FuncPropType Ana;
      FuncPropType::Result Result = Ana.run(IR, FAM);
      return new PredictorInput{Result, FAM.PassResults[&IR], PassName};
    }
#endif
  return nullptr;
}

} // namespace llvm
