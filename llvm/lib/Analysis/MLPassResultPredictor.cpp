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
static llvm::cl::opt<unsigned> PredictionReuse("prediction-reuse", cl::Hidden,
                                               cl::desc("reuse-prediction."),
                                               cl::init(4));
static cl::opt<bool> RunPrediction("run-prediction", cl::Hidden,
                                   cl::desc("pass result predictor"),
                                   cl::init(false));
static cl::opt<bool> RunBatchPrediction("run-batch-prediction", cl::Hidden,
                                        cl::desc("pass result predictor"),
                                        cl::init(false));

static cl::opt<bool> DumpBatch("dump-batch", cl::Hidden,
                               cl::desc("pass result predictor"),
                               cl::init(false));

static cl::opt<bool> DumpAllResult("dump-all-result", cl::Hidden,
                                   cl::desc("dump pass results"),
                                   cl::init(false));
static cl::opt<bool> NotRunForTrivial(
    "not-run-for-trivial", cl::Hidden,
    cl::desc(
        "this flag makes predicotr not run for small function (instruction<5)"),
    cl::init(false));

static cl::opt<PredictionMethod> PMethod(
    "prediction-method", cl::desc("Choose Prediction Method:"),
    cl::init(BatchNN),
    cl::values(clEnumVal(SingleNN, "Single pass prediction with NN"),
               clEnumVal(SingleLogistic,
                         "Single pass prediction with logistic regression"),
               clEnumVal(BatchNN, "Batch pass prediction with NN"),
               clEnumVal(Sequential, "Sequential model with pass dep")));

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
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dumpAllResult(
    Function &IR, FunctionAnalysisManager &FAM) {
  if (!DumpAllResult)
    return;

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

  std::vector<int> Results;

  for (const auto &Pass : FPM.Passes) {
    ValueToValueMapTy VMap;
    Function &F = *CloneFunction(&IR, VMap);
    FAM.clear();
    PreservedAnalyses Res = Pass->run(F, FAM);
    Results.push_back(!Res.areAllPreserved());
    VMap.clear();
    F.eraseFromParent();
  }

  LLVM_DEBUG({
    dbgs() << IR.getName() << " ";
    for (size_t i = 0; i < Results.size(); i++) {
      dbgs() << FPM.Passes[i]->name() << ",";
    }
    dbgs() << "\n";
  });

  FunctionPropertiesAnalysis::Result FPI =
      FAM.getResult<FunctionPropertiesAnalysis>(IR);
  auto CodeFeature = FPI.toVec();

  {
    // Logging to temp file.
    int pid = getpid();
    auto fname = "pass_all_data_" + std::to_string(pid) + ".txt";
    std::error_code EC;
    auto os = std::make_unique<llvm::raw_fd_ostream>(fname, EC,
                                                     llvm::sys::fs::OF_Append);
    *os << IR.getName() << ",";
    for (auto s : CodeFeature) {
      *os << s << ",";
    }
    for (auto s : Results) {
      *os << s << ",";
    }
    *os << "\n";
  }
}

template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dumpAllResult(
    Module &IR, ModuleAnalysisManager &MAM) {}

template <>
bool MLPassResultPredictor<Module, ModuleAnalysisManager>::predictPassResult(
    Module &In, ModuleAnalysisManager &MAM, StringRef name) {
  // For now, we don't predict Module pass results.
  return true;
}
bool predictPassResultByAllModel(Function &F, FunctionAnalysisManager &FAM,
                                 StringRef name, bool &ActualRun) {
  // Dirty hack.
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

  for (size_t i = 0; i < FPM.Passes.size(); i++) {
    if (FPM.Passes[i]->name() == name) {
      bool res = true;
      {
        FunctionPassResultAnalysis::Result *FPP =
            FAM.getCachedResult<FunctionPassResultAnalysis>(F);
        if (FPP) {
          LLVM_DEBUG(dbgs() << FPP->counter << " counter\n";);
          FPP->counter++;
          if (FPP->counter < PredictionReuse)
            res = FPP->result[i];
          else
            res = FAM.getResult<FunctionPassResultAnalysis>(F).result[i];
        } else {
          res = FAM.getResult<FunctionPassResultAnalysis>(F).result[i];
        }
      }
      ActualRun = true;
      return res;
    }
  }
  return true;
}

#include "./Prediction/predictor_CorrelatedValuePropagationPass.inc"
#include "./Prediction/predictor_GVN.inc"
#include "./Prediction/predictor_InstSimplifyPass.inc"
#include "./Prediction/predictor_JumpThreadingPass.inc"
#include "./Prediction/predictor_LoopUnrollPass.inc"
#include "./Prediction/predictor_ReassociatePass.inc"
#include "./Prediction/predictor_SROA.inc"
#include "./Prediction/predictor_SimplifyCFGPass.inc"

bool predictPassResultBySingleLogistic(Function &F,
                                       FunctionAnalysisManager &FAM,
                                       StringRef PassName, bool &ActualRun) {
#define CHECK(NAME)                                                            \
  {                                                                            \
    if (PassName == #NAME) {                                                   \
      FunctionPropertiesAnalysis::Result FPI =                                 \
          FAM.getResult<FunctionPropertiesAnalysis>(F);                        \
      ActualRun = true;                                                        \
      return predict_##NAME(FPI);                                              \
    }                                                                          \
  }
  CHECK(ReassociatePass);
  CHECK(InstSimplifyPass);
  CHECK(CorrelatedValuePropagationPass);
  CHECK(JumpThreadingPass);
  CHECK(LoopUnrollPass);
  CHECK(SimplifyCFGPass);
  CHECK(SROA);
  CHECK(GVN);
#undef CHECK
  return true;
}

bool predictPassResultBySingleNN(Function &F, FunctionAnalysisManager &FAM,
                                 StringRef PassName, bool &ActualRun) {
  auto &Ctx = F.getContext();
  auto Predictor = Ctx.getPredictor();
  if (!Predictor) {
    Ctx.setPredictor(std::make_unique<FunctionPassResultPredictionModel>());
    Predictor = Ctx.getPredictor();
  }
  assert(Predictor && "Never null");
  auto Model = Predictor->get(PassName.str(), Ctx);
  if (!Model) {
    ActualRun = false;
    return true;
  }

  FunctionPropertiesAnalysis::Result FPI =
      FAM.getResult<FunctionPropertiesAnalysis>(F);
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
  return Model->run();
}
template <>
bool MLPassResultPredictor<Function, FunctionAnalysisManager>::
    predictPassResult(Function &F, FunctionAnalysisManager &FAM,
                      StringRef Name) {
  NumPassesRun++;
  if (!RunPrediction)
    return true;

  if (NotRunForTrivial && F.getInstructionCount() < 5) {
    NumPredictionForTrivial++;
    return true;
  }

  bool ActualRun = false;
  bool Result = true;

  switch (PMethod) {
  case PredictionMethod::BatchNN:
    Result = predictPassResultByAllModel(F, FAM, Name, ActualRun);
    break;
  case PredictionMethod::SingleLogistic:
    Result = predictPassResultBySingleLogistic(F, FAM, Name, ActualRun);
    break;
  case PredictionMethod::SingleNN:
    Result = predictPassResultBySingleNN(F, FAM, Name, ActualRun);
    break;
  }
  if (ActualRun) {
    (Result ? NumPredictionTrue : NumPredictionFalse)++;
    NumPrediction++;
  }
  return Result;
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
    updatePassResults(std::vector<StringRef> &names, Function &F,
                      FunctionAnalysisManager &FAM,
                      Optional<std::vector<bool>> &res, int index,
                      std::vector<bool> &previous_result) {

  TimeTraceScope TimeScope("Prediction time", F.getName());
  if (!res || names.size() < 30)
    return;

  if (F.getInstructionCount() < 5) { // Don't call for small function
    return;
  }

  if (index == 5 || index == 14 || index == 16 || index == 26 || index == 29) {
    // CorrelatedValueProp -> SimplifyCFG
    res.getValue()[index] = previous_result.back();
    return;
  }

  if (index == 6 || index == 12 || index == 18 || index == 24 || index == 30) {
    // predict next 6 pass
    FunctionPropertiesAnalysis::Result FPI =
        FAM.getResult<FunctionPropertiesAnalysis>(F);
    auto CodeFeature = FPI.toVec();
    if (DumpBatch) {
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
      NumBatch++;
      if (index == 30)
        return;
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
          previous_result[previous_result.size() - i - 1];                     \
    }                                                                          \
    model##NAME->Run();                                                        \
    NumPrediction++;                                                           \
    for (int i = 0; i < 6; i++) {                                              \
      res.getValue()[index + i] = model##NAME->result0(0, i) > 0.5;            \
      (res.getValue()[index + i] ? NumPredictionTrue : NumPredictionFalse)++;  \
    }                                                                          \
  }

#define CHECK2(NAME)                                                            \
  if (index == NAME) {                                                         \
    if (!model##NAME) {                                                        \
      modelO2_##NAME = std::make_unique<ModelO2_##NAME>();                           \
    }                                                                          \
    for (int i = 0; i < CodeFeature.size(); i++) {                             \
      modelO2_##NAME->arg_feed_Input(0, i) = CodeFeature[i];                      \
    }                                                                          \
    for (int i = 0; i < 6; i++) {                                              \
      modelO2_##NAME->arg_feed_Input(0, i + CodeFeature.size()) =                 \
          previous_result[previous_result.size() - i - 1];                     \
    }                                                                          \
    modelO2_##NAME->Run();                                                        \
    NumPrediction++;                                                           \
    for (int i = 0; i < 6; i++) {                                              \
      res.getValue()[index + i] = modelO2_##NAME->result0(0, i) > 0.5;            \
      (res.getValue()[index + i] ? NumPredictionTrue : NumPredictionFalse)++;  \
    }                                                                          \
  }
      if (names.size() == 31) {
        CHECK(6);
        CHECK(12);
        CHECK(18);
        CHECK(24);
      } else if (names.size() == 30) {
        CHECK2(6);
        CHECK2(12);
        CHECK2(18);
        CHECK2(24);
      }

      LLVM_DEBUG({
        dbgs() << F.getName() << " " << index << "\n";
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
    std::vector<StringRef> &names, Module &In, ModuleAnalysisManager &MAM,
    Optional<std::vector<bool>> &res, int index,
    std::vector<bool> &previous_result) {
  if (index + 1 == names.size() && DumpBatch) {
    auto os = pid_logger("batch8");
    *os << names[0] << "\t";
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
    std::vector<StringRef> &names, Function &In, FunctionAnalysisManager &MAM) {
  if (names.size() < 30 || In.getInstructionCount() < 5)
    return None;

  // auto os = pid_logger("batch6");
  // *os << "start\n";
  std::vector<bool> res(names.size(), true);
  return res;
}
template <>
Optional<std::vector<bool>>
MLPassResultPredictor<Module, ModuleAnalysisManager>::predictPassResults(
    std::vector<StringRef> &names, Module &In, ModuleAnalysisManager &MAM) {
  // For now, we don't predict Module pass results.

  // if (names.size() == 9 && DumpBatch) {
  //   std::vector<bool> res(9, true);
  //   dbgs() << names.size() << "\n";
  //   for (auto s : names)
  //     dbgs() << s << " ";
  //   dbgs() << "\n";
  //   return res;
  // }

  return None;
}
template <>
void MLPassResultPredictor<Module, ModuleAnalysisManager>::dumpAfterPasses(
    std::vector<StringRef> &names, Module &In, ModuleAnalysisManager &MAM,
    std::vector<bool> &res) {}
template <>
void MLPassResultPredictor<Function, FunctionAnalysisManager>::dumpAfterPasses(
    std::vector<StringRef> &names, Function &In, FunctionAnalysisManager &MAM,
    std::vector<bool> &res) {
  if (DumpBatch) {
    auto os = pid_logger("dep");
    *os << "start"
        << "\n";
    for (int i = 0; i < names.size(); i++) {
      *os << i << "\t" << names[i] << "\t" << res[i] << "\n";
    }
    *os << "end"
        << "\n";
    if (res.size() == 30) {
      auto os = pid_logger("batch7");
      for (int i = res.size() - 12; i < res.size(); i++) {
        *os << res[i] << "\t";
      }
      *os << "\n";
    }
  }
}

} // namespace llvm
