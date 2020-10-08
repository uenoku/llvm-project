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
#include <sys/types.h>
#include <unistd.h>

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/MLPassResultPredictor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "ModelO2_12.h"
#include "ModelO2_18.h"
#include "ModelO2_24.h"
#include "ModelO2_6.h"
#include "ModelO3_12.h"
#include "ModelO3_18.h"
#include "ModelO3_24.h"
#include "ModelO3_6.h"
#define DEBUG_TYPE "pass-result-predictor-ml"

namespace llvm {
static cl::opt<bool> RunPassSkip("run-pass-skip", cl::Hidden,
                                        cl::desc("Run a prediction model"),
                                        cl::init(false));

static cl::opt<bool> DumpDataSet("dump-data-set", cl::Hidden,
                               cl::desc("Dump data set if true"),
                               cl::init(false));

static cl::opt<float> Threshold("prediction-threshold", cl::Hidden,
                                cl::desc("Threshold moving parameter"), cl::init(0.5));
static cl::opt<unsigned int> MinimumFunctionSizeThreshold("minimum-function-size-threshold", cl::Hidden,
                                   cl::desc("Call prediction models only for functions whose size are greater than this"), cl::init(5));

STATISTIC(NumPredictionForTrivial, "Number of prediction for prediction");
STATISTIC(NumPrediction, "Number of prediction");
STATISTIC(NumPassesRun, "Number of run of passes");
STATISTIC(NumPredictionTrue, "Number of prediction true");
STATISTIC(NumPredictionFalse, "Number of prediction false");

std::unique_ptr<llvm::raw_fd_ostream> pidLogger(std::string Prefix) {
  // Logging to temp file.

  int pid = getpid();
  auto fname = Prefix + std::to_string(pid) + ".txt";
  std::error_code EC;
  auto os = std::make_unique<llvm::raw_fd_ostream>(fname, EC,
                                                   llvm::sys::fs::OF_Append);
  return os;
}

namespace Model {
// Predictors
// FIXME: Move these static variables to LLVM Context as soon as possible.
static std::unique_ptr<ModelO3_6> O3_6;
static std::unique_ptr<ModelO3_12> O3_12;
static std::unique_ptr<ModelO3_18> O3_18;
static std::unique_ptr<ModelO3_24> O3_24;
static std::unique_ptr<ModelO2_6> O2_6;
static std::unique_ptr<ModelO2_12> O2_12;
static std::unique_ptr<ModelO2_18> O2_18;
static std::unique_ptr<ModelO2_24> O2_24;
} // namespace Model

template <typename ModelTy>
void invokeModelAndUpdateResult(std::unique_ptr<ModelTy> &Model, Optional<std::vector<bool>> &Result,
         int Index, std::vector<bool> &PreviousResult,
         std::vector<int64_t> &CodeFeature) {
  if (!Model) {
    Model = std::make_unique<ModelTy>();
  }
  for (unsigned int I = 0; I < CodeFeature.size(); I++) {
    Model->arg_feed_Input(0, I) = CodeFeature[I];
  }
  for (unsigned int I = 0; I < 6; I++) {
    Model->arg_feed_Input(0, I + CodeFeature.size()) =
        PreviousResult[PreviousResult.size() + I - 6];
  }
  Model->Run();
  NumPrediction++;

  for (unsigned int I = 0; I < 6; I++) {
    float RawProb = Model->result0(0, I);
    LLVM_DEBUG({
      dbgs() << "\nPassPrediction Raw Value" << Index << " " << RawProb << "\n";
    });
    Result.getValue()[Index + I] = RawProb > Threshold;
    (Result.getValue()[Index + I] ? NumPredictionTrue : NumPredictionFalse)++;
  }
}

template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::
    updatePassResultsPrediction(int LengthOfPassPipeline, Function &F,
                      FunctionAnalysisManager &FAM,
                      Optional<std::vector<bool>> &Result, int Index,
                      std::vector<bool> &PreviousResult) {
  // FIXME: Bad hack
  if (!Result || LengthOfPassPipeline < 30)
    return;

  if (F.getInstructionCount() <
      MinimumFunctionSizeThreshold) { // Don't call for small functions
    return;
  }

  if (Index && Index % 6 == 0) {
    // Dump or run models only when Index % 6 == 0
    if (DumpDataSet) {
      FunctionPropertiesAnalysis::Result FPI =
          FAM.getResult<FunctionPropertiesAnalysis>(F);

      auto CodeFeature = FPI.toVec();
      auto OS = pidLogger("batch");

      if (Index != 6) {
        for (unsigned int I = PreviousResult.size() - 12; I < PreviousResult.size();
             I++) {
          *OS << PreviousResult[I] << "\t";
        }
        *OS << "\n";
      } else {
        *OS << "\n";
      }
      if (Index != 30) {
        *OS << Index << "\t";
        for (unsigned int I = 0; I < CodeFeature.size(); I++) {
          *OS << CodeFeature[I] << "\t";
        }
      }
    } else if (RunPassSkip) {
      if (Index == 30)
        return;
      TimeTraceScope TimeScope("Prediction time", F.getName());
      FunctionPropertiesAnalysis::Result FPI =
          FAM.getResult<FunctionPropertiesAnalysis>(F);
      auto CodeFeature = FPI.toVec();

      // FIXME: Avoid macro ....
#define RUN_O3(NAME)                                                           \
  if (Index == NAME)                                                           \
    invokeModelAndUpdateResult(Model::O3_##NAME, Result, Index, PreviousResult, CodeFeature);
#define RUN_O2(NAME)                                                           \
  if (Index == NAME)                                                           \
    invokeModelAndUpdateResult(Model::O2_##NAME, Result, Index, PreviousResult, CodeFeature);

  // FIXME: Currently, we judge the pipeline by its length. This is very
  // fragile.
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
        dbgs() << "\nPassPrediction " << F.getName() << " " << Index << "\n";
        for (int i = 0; i < 6; i++) {
          dbgs() << Result.getValue()[Index + i] << " ";
        }
        dbgs() << "\n";
      });

#undef RUN_O3
#undef RUN_O2
    }
  }

  return;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::updatePassResultsPrediction(
    int LengthOfPassPipeline, Module &M, ModuleAnalysisManager &MAM,
    Optional<std::vector<bool>> &Result, int Index,
    std::vector<bool> &PreviousResult) {
  // TODO: Support for Module passes
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Function, FunctionAnalysisManager>::initializePassResultsPrediction(
    int LengthOfPassPipeline, Function &F, FunctionAnalysisManager &MAM) {
  // FIXME: Currently, we judge the pipeline by its length. This is very
  // fragile.
  if (LengthOfPassPipeline < 30 || F.getInstructionCount() < MinimumFunctionSizeThreshold)
    return None;
  std::vector<bool> Res(LengthOfPassPipeline, true);
  return Res;
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Module, ModuleAnalysisManager>::initializePassResultsPrediction(
    int LengthOfPassPipeline, Module &In, ModuleAnalysisManager &MAM) {
  // For now, we don't predict Module pass results.

  return None;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dumpAfterPasses(
    int LengthOfPassPipeline, Module &M, ModuleAnalysisManager &MAM,
    std::vector<bool> &res) {}
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dumpAfterPasses(
    int LengthOfPassPipeline, Function &F, FunctionAnalysisManager &MAM,
    std::vector<bool> &Result) {}

template <> bool MLPassResultPredictor<Module, ModuleAnalysisManager>::valid() {
  return DumpDataSet || RunPassSkip;
}
template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::valid() {
  return DumpDataSet || RunPassSkip;
}
} // namespace llvm
