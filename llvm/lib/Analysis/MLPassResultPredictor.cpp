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
// Predictors
// FIXME: Move these static variables to LLVM Context as soon as possible.
static std::unique_ptr<Model6> modelO3_6;
static std::unique_ptr<Model12> modelO3_12;
static std::unique_ptr<Model18> modelO3_18;
static std::unique_ptr<Model24> modelO3_24;
static std::unique_ptr<ModelO2_6> modelO2_6;
static std::unique_ptr<ModelO2_12> modelO2_12;
static std::unique_ptr<ModelO2_18> modelO2_18;
static std::unique_ptr<ModelO2_24> modelO2_24;

template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::
    updatePassResults(int LengthOfPassPipeline, Function &F,
                      FunctionAnalysisManager &FAM,
                      Optional<std::vector<bool>> &Result, int index,
                      std::vector<bool> &PreviousResult) {
  // FIXME: Bad hack
  if (!Result || LengthOfPassPipeline < 30)
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
      auto os = pid_logger("batch");

      if (index != 6) {
        for (int i = PreviousResult.size() - 12; i < PreviousResult.size();
             i++) {
          *os << PreviousResult[i] << "\t";
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
      }
    } else if (RunBatchPrediction) {
      if (index == 30)
        return;
      TimeTraceScope TimeScope("Prediction time", F.getName());
      FunctionPropertiesAnalysis::Result FPI =
          FAM.getResult<FunctionPropertiesAnalysis>(F);
      auto CodeFeature = FPI.toVec();

      // FIXME: Avoid macro ....
#define RUN_O3(NAME)                                                           \
  if (index == NAME) {                                                         \
    if (!modelO3_##NAME) {                                                     \
      modelO3_##NAME = std::make_unique<Model##NAME>();                        \
    }                                                                          \
    for (int i = 0; i < CodeFeature.size(); i++) {                             \
      modelO3_##NAME->arg_feed_Input(0, i) = CodeFeature[i];                   \
    }                                                                          \
    for (int i = 0; i < 6; i++) {                                              \
      modelO3_##NAME->arg_feed_Input(0, i + CodeFeature.size()) =              \
          PreviousResult[PreviousResult.size() + i - 6];                       \
    }                                                                          \
    modelO3_##NAME->Run();                                                     \
    NumPrediction++;                                                           \
    for (int i = 0; i < 6; i++) {                                              \
      Result.getValue()[index + i] =                                           \
          f(modelO3_##NAME->result0(0, i)) > threshold;                        \
      (Result.getValue()[index + i] ? NumPredictionTrue                        \
                                    : NumPredictionFalse)++;                   \
    }                                                                          \
  }

#define RUN_O2(NAME)                                                           \
  if (index == NAME) {                                                         \
    if (!modelO2_##NAME) {                                                     \
      modelO2_##NAME = std::make_unique<ModelO2_##NAME>();                     \
    }                                                                          \
    for (int i = 0; i < CodeFeature.size(); i++) {                             \
      modelO2_##NAME->arg_feed_Input(0, i) = CodeFeature[i];                   \
    }                                                                          \
    for (int i = 0; i < 6; i++) {                                              \
      modelO2_##NAME->arg_feed_Input(0, i + CodeFeature.size()) =              \
          PreviousResult[PreviousResult.size() - 6 + i];                       \
    }                                                                          \
    modelO2_##NAME->Run();                                                     \
    NumPrediction++;                                                           \
    for (int i = 0; i < 6; i++) {                                              \
      Result.getValue()[index + i] =                                           \
          f(modelO2_##NAME->result0(0, i)) > threshold;                        \
      (Result.getValue()[index + i] ? NumPredictionTrue                        \
                                    : NumPredictionFalse)++;                   \
    }                                                                          \
  }
      auto f = [&](float v) {
        LLVM_DEBUG({
          dbgs() << "\nPassPrediction Raw Value" << index << " " << v << "\n";
        });
        return v;
      };
      if (LengthOfPassPipeline == 31) {
        RUN_O3(6);
        RUN_O3(12);
        RUN_O3(18);
        RUN_O3(24);
      } else if (LengthOfPassPipeline == 30) {
        RUN_O2(6);
        RUN_O2(12);
        RUN_O2(18);
        RUN_O2(24);
      }

      LLVM_DEBUG({
        dbgs() << "\nPassPrediction " << F.getName() << " " << index << "\n";
        for (int i = 0; i < 6; i++) {
          dbgs() << Result.getValue()[index + i] << " ";
        }
        dbgs() << "\n";
      });
    }
  }
#undef RUN_O2
#undef RUN_O3

  return;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::updatePassResults(
    int LengthOfPassPipeline, Module &M, ModuleAnalysisManager &MAM,
    Optional<std::vector<bool>> &Result, int Index,
    std::vector<bool> &PreviousResult) {
  // TODO: Support for Module passes
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Function, FunctionAnalysisManager>::predictPassResults(
    int LengthOfPassPipeline, Function &F, FunctionAnalysisManager &MAM) {
  // FIXME: Currently, we judge the pipeline by its length. This is very
  // fragile.
  if (LengthOfPassPipeline < 30 || F.getInstructionCount() < size_threshold)
    return None;
  std::vector<bool> Res(LengthOfPassPipeline, true);
  return Res;
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Module, ModuleAnalysisManager>::predictPassResults(
    int LengthOfPassPipeline, Module &In, ModuleAnalysisManager &MAM) {
  // For now, we don't predict Module pass results.

  return None;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dumpAfterPasses(
    int LengthOfPassPipeline, Module &M, ModuleAnalysisManager &MAM, std::vector<bool> &res) {
}
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dumpAfterPasses(
    int LengthOfPassPipeline, Function &F, FunctionAnalysisManager &MAM,
    std::vector<bool> &Result) {}
template <> bool MLPassResultPredictor<Module, ModuleAnalysisManager>::valid() {
  return DumpBatch || RunBatchPrediction;
}
template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::valid() {
  return DumpBatch || RunBatchPrediction;
}
} // namespace llvm
