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

#include "Model12.h"
#include "Model18.h"
#include "Model24.h"
#include "Model6.h"
#include "ModelO2_12.h"
#include "ModelO2_18.h"
#include "ModelO2_24.h"
#include "ModelO2_6.h"
#define DEBUG_TYPE "pass-result-predictor-ml"

namespace llvm {
static cl::opt<bool> RunBatchPrediction("run-batch-prediction", cl::Hidden,
                                        cl::desc("pass result predictor"),
                                        cl::init(false));

static cl::opt<bool> DumpBatch("dump-batch", cl::Hidden,
                               cl::desc("pass result predictor"),
                               cl::init(false));

static cl::opt<float> threshold("prediction-threshold", cl::Hidden,
                                cl::desc("dump pass results"), cl::init(0.5));
static cl::opt<int> size_threshold("size_threshold", cl::Hidden,
                                   cl::desc("dump pass results"), cl::init(5));

STATISTIC(NumPredictionForTrivial, "Number of prediction for prediction");
STATISTIC(NumPrediction, "Number of prediction");
STATISTIC(NumPassesRun, "Number of run of passes");
STATISTIC(NumPredictionTrue, "Number of prediction true");
STATISTIC(NumPredictionFalse, "Number of prediction false");
STATISTIC(NumBatch, "Show that batch mode");
std::unique_ptr<llvm::raw_fd_ostream> pid_logger(std::string prefix) {
  // Logging to temp file.
  int pid = getpid();
  auto fname = prefix + std::to_string(pid) + ".txt";
  std::error_code EC;
  auto os = std::make_unique<llvm::raw_fd_ostream>(fname, EC,
                                                   llvm::sys::fs::OF_Append);
  return os;
}

static std::unique_ptr<Model6> model6;
static std::unique_ptr<Model12> model12;
static std::unique_ptr<Model18> model18;
static std::unique_ptr<Model24> model24;

static std::unique_ptr<ModelO2_6> modelO2_6;
static std::unique_ptr<ModelO2_12> modelO2_12;
static std::unique_ptr<ModelO2_18> modelO2_18;
static std::unique_ptr<ModelO2_24> modelO2_24;


template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::
    updatePassResults(int names, Function &F, FunctionAnalysisManager &FAM,
                      Optional<std::vector<bool>> &res, int index,
                      std::vector<bool> &previous_result) {

  if (!res || names < 30)
    return;

  if (F.getInstructionCount() <
      size_threshold) { // Don't call for small function
    return;
  }

  if (index && index % 6 == 0) {
    // predict next 6 pass

    if (DumpBatch) {

      FunctionPropertiesAnalysis::Result FPI =
          FAM.getResult<FunctionPropertiesAnalysis>(F);

      auto CodeFeature = FPI.toVec();
      auto os = pid_logger("batch7");

      if (index != 6) {
        for (int i = previous_result.size() - 12; i < previous_result.size();
             i++) {
          *os << previous_result[i] << "\t";
        }
        *os << "\n";
      } else {
        *os << "\n";
      }
      if (index != 30) {
        *os << index << "\t";
        for (int i = 0; i < CodeFeature.size(); i++) {
          *os << CodeFeature[i] << "\t";
        }
        // *os << "\n";
      }
    } else if (RunBatchPrediction) {
      if (index == 30)
        return;
      TimeTraceScope TimeScope("Prediction time", F.getName());
      FunctionPropertiesAnalysis::Result FPI =
          FAM.getResult<FunctionPropertiesAnalysis>(F);
      auto CodeFeature = FPI.toVec();

      // FIXME: Avoid macro ....
#define CHECK(NAME)                                                            \
  if (index == NAME) {                                                         \
    if (!model##NAME) {                                                        \
      model##NAME = std::make_unique<Model##NAME>();                           \
    }                                                                          \
    for (int i = 0; i < CodeFeature.size(); i++) {                             \
      model##NAME->arg_feed_Input(0, i) = CodeFeature[i];                      \
    }                                                                          \
    for (int i = 0; i < 6; i++) {                                              \
      model##NAME->arg_feed_Input(0, i + CodeFeature.size()) =                 \
          previous_result[previous_result.size() + i - 6];                     \
    }                                                                          \
    model##NAME->Run();                                                        \
    NumPrediction++;                                                           \
    for (int i = 0; i < 6; i++) {                                              \
      res.getValue()[index + i] = f(model##NAME->result0(0, i)) > threshold;   \
      (res.getValue()[index + i] ? NumPredictionTrue : NumPredictionFalse)++;  \
    }                                                                          \
  }

#define CHECK2(NAME)                                                           \
  if (index == NAME) {                                                         \
    if (!modelO2_##NAME) {                                                     \
      modelO2_##NAME = std::make_unique<ModelO2_##NAME>();                     \
    }                                                                          \
    for (int i = 0; i < CodeFeature.size(); i++) {                             \
      modelO2_##NAME->arg_feed_Input(0, i) = CodeFeature[i];                   \
    }                                                                          \
    for (int i = 0; i < 6; i++) {                                              \
      modelO2_##NAME->arg_feed_Input(0, i + CodeFeature.size()) =              \
          previous_result[previous_result.size() - 6 + i];                     \
    }                                                                          \
    modelO2_##NAME->Run();                                                     \
    NumPrediction++;                                                           \
    for (int i = 0; i < 6; i++) {                                              \
      res.getValue()[index + i] =                                              \
          f(modelO2_##NAME->result0(0, i)) > threshold;                        \
      (res.getValue()[index + i] ? NumPredictionTrue : NumPredictionFalse)++;  \
    }                                                                          \
  }
      auto f = [&](float v) {
        LLVM_DEBUG({
          dbgs() << "\nPassPrediction Raw Value" << index << " " << v << "\n";
        });
        return v;
      };
      if (names == 31) {
        CHECK(6);
        CHECK(12);
        CHECK(18);
        CHECK(24);
      } else if (names == 30) {
        CHECK2(6);
        CHECK2(12);
        CHECK2(18);
        CHECK2(24);
      }

      LLVM_DEBUG({
        dbgs() << "\nPassPrediction " << F.getName() << " " << index << "\n";
        for (int i = 0; i < 6; i++) {
          dbgs() << res.getValue()[index + i] << " ";
        }
        dbgs() << "\n";
      });
    }
  }
#undef CHECK
#undef CHECK2

  return;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::updatePassResults(
    int names, Module &In, ModuleAnalysisManager &MAM,
    Optional<std::vector<bool>> &res, int index,
    std::vector<bool> &previous_result) {
  if (index + 1 == names && DumpBatch) {
    auto os = pid_logger("batch8");
    //    *os << names[0] << "\t";
    for (int i = 0; i < previous_result.size(); i++) {
      *os << previous_result[i] << "\t";
    }
    *os << "\n";
  }

  return;
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Function, FunctionAnalysisManager>::predictPassResults(
    int names, Function &In, FunctionAnalysisManager &MAM) {
  // FIXME: Currently, we judge the pipeline by its length. This is very fragile.
  if (names < 30 || In.getInstructionCount() < size_threshold)
    return None;
  std::vector<bool> res(names, true);
  return res;
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Module, ModuleAnalysisManager>::predictPassResults(
    int names, Module &In, ModuleAnalysisManager &MAM) {
  // For now, we don't predict Module pass results.

  return None;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dumpAfterPasses(
    int names, Module &In, ModuleAnalysisManager &MAM, std::vector<bool> &res) {
}
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dumpAfterPasses(
    int names, Function &In, FunctionAnalysisManager &MAM,
    std::vector<bool> &res) {
}
template <> bool MLPassResultPredictor<Module, ModuleAnalysisManager>::valid() {
  return DumpBatch || RunBatchPrediction;
}
template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::valid() {
  return DumpBatch || RunBatchPrediction;
}
} // namespace llvm
