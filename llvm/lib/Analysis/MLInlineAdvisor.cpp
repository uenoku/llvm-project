//===- MLInlineAdvisor.cpp - machine learned InlineAdvisor ----------------===//
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
#include "llvm/Config/config.h"
#if defined(LLVM_HAVE_TF_AOT) || defined(LLVM_HAVE_TF_API)

#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/MLInlineAdvisor.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

using namespace llvm;

#define DEBUG_TYPE "inline-ml"

static cl::opt<float> SizeIncreaseThreshold(
    "ml-advisor-size-increase-threshold", cl::Hidden,
    cl::desc("Maximum factor by which expected native size may increase before "
             "blocking any further inlining."),
    cl::init(2.0));

const std::array<std::string, NumberOfFeatures> llvm::FeatureNameMap{
#define POPULATE_NAMES(INDEX_NAME, NAME, COMMENT) NAME,
    INLINE_FEATURE_ITERATOR(POPULATE_NAMES)
#undef POPULATE_NAMES
};

const char *const llvm::DecisionName = "inlining_decision";
const char *const llvm::DefaultDecisionName = "inlining_default";
const char *const llvm::RewardName = "delta_size";

CallBase *getInlinableCS(Instruction &I) {
  if (auto *CS = dyn_cast<CallBase>(&I))
    if (Function *Callee = CS->getCalledFunction()) {
      if (!Callee->isDeclaration()) {
        return CS;
      }
    }
  return nullptr;
}

MLInlineAdvisor::MLInlineAdvisor(Module &M, ModuleAnalysisManager &MAM,
                                 std::unique_ptr<MLModelRunner> Runner)
    : InlineAdvisor(
          MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
      M(M), ModelRunner(std::move(Runner)), CG(new CallGraph(M)),
      InitialIRSize(getModuleIRSize()), CurrentIRSize(InitialIRSize) {
  assert(ModelRunner);

  // Extract the 'call site height' feature - the position of a call site
  // relative to the farthest statically reachable SCC node. We don't mutate
  // this value while inlining happens. Empirically, this feature proved
  // critical in behavioral cloning - i.e. training a model to mimic the manual
  // heuristic's decisions - and, thus, equally important for training for
  // improvement.
  for (auto I = scc_begin(CG.get()); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &CGNodes = *I;
    unsigned Level = 0;
    for (auto *CGNode : CGNodes) {
      Function *F = CGNode->getFunction();
      if (!F || F->isDeclaration())
        continue;
      for (auto &I : instructions(F)) {
        if (auto *CS = getInlinableCS(I)) {
          auto *Called = CS->getCalledFunction();
          auto Pos = FunctionLevels.find(Called);
          // In bottom up traversal, an inlinable callee is either in the
          // same SCC, or to a function in a visited SCC. So not finding its
          // level means we haven't visited it yet, meaning it's in this SCC.
          if (Pos == FunctionLevels.end())
            continue;
          Level = std::max(Level, Pos->second + 1);
        }
      }
    }
    for (auto *CGNode : CGNodes) {
      Function *F = CGNode->getFunction();
      if (F && !F->isDeclaration())
        FunctionLevels[F] = Level;
    }
  }
}

void MLInlineAdvisor::onPassEntry() {
  // Function passes executed between InlinerPass runs may have changed the
  // module-wide features.
  NodeCount = 0;
  EdgeCount = 0;
  for (auto &F : M)
    if (!F.isDeclaration()) {
      ++NodeCount;
      EdgeCount += getLocalCalls(F);
    }
}

int64_t MLInlineAdvisor::getLocalCalls(Function &F) {
  return FAM.getResult<FunctionPropertiesAnalysis>(F)
      .DirectCallsToDefinedFunctions;
}

// Update the internal state of the advisor, and force invalidate feature
// analysis. Currently, we maintain minimal (and very simple) global state - the
// number of functions and the number of static calls. We also keep track of the
// total IR size in this module, to stop misbehaving policies at a certain bloat
// factor (SizeIncreaseThreshold)
void MLInlineAdvisor::onSuccessfulInlining(const MLInlineAdvice &Advice,
                                           bool CalleeWasDeleted) {
  assert(!ForceStop);
  Function *Caller = Advice.getCaller();
  Function *Callee = Advice.getCallee();

  // The caller features aren't valid anymore.
  FAM.invalidate<FunctionPropertiesAnalysis>(*Caller);
  int64_t IRSizeAfter =
      getIRSize(*Caller) + (CalleeWasDeleted ? 0 : Advice.CalleeIRSize);
  CurrentIRSize += IRSizeAfter - (Advice.CallerIRSize + Advice.CalleeIRSize);
  if (CurrentIRSize > SizeIncreaseThreshold * InitialIRSize)
    ForceStop = true;

  // We can delta-update module-wide features. We know the inlining only changed
  // the caller, and maybe the callee (by deleting the latter).
  // Nodes are simple to update.
  // For edges, we 'forget' the edges that the caller and callee used to have
  // before inlining, and add back what they currently have together.
  int64_t NewCallerAndCalleeEdges =
      FAM.getResult<FunctionPropertiesAnalysis>(*Caller)
          .DirectCallsToDefinedFunctions;

  if (CalleeWasDeleted)
    --NodeCount;
  else
    NewCallerAndCalleeEdges +=
        FAM.getResult<FunctionPropertiesAnalysis>(*Callee)
            .DirectCallsToDefinedFunctions;
  EdgeCount += (NewCallerAndCalleeEdges - Advice.CallerAndCalleeEdges);
  assert(CurrentIRSize >= 0 && EdgeCount >= 0 && NodeCount >= 0);
}

int64_t MLInlineAdvisor::getModuleIRSize() const {
  int64_t Ret = 0;
  for (auto &F : CG->getModule())
    if (!F.isDeclaration())
      Ret += getIRSize(F);
  return Ret;
}

std::unique_ptr<InlineAdvice> MLInlineAdvisor::getAdvice(CallBase &CB) {
  auto &Caller = *CB.getCaller();
  auto &Callee = *CB.getCalledFunction();

  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto GetTLI = [&](Function &F) -> const TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };

  auto &TIR = FAM.getResult<TargetIRAnalysis>(Callee);
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(Caller);

  auto TrivialDecision =
      llvm::getAttributeBasedInliningDecision(CB, &Callee, TIR, GetTLI);

  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same thing if this is a recursive case.
  if ((TrivialDecision.hasValue() && !TrivialDecision->isSuccess()) ||
      &Caller == &Callee)
    return std::make_unique<InlineAdvice>(this, CB, ORE, false);

  bool Mandatory = TrivialDecision.hasValue() && TrivialDecision->isSuccess();

  // If we need to stop, we won't want to track anymore any state changes, so
  // we just return the base InlineAdvice, which acts as a noop.
  if (ForceStop) {
    ORE.emit([&] {
      return OptimizationRemarkMissed(DEBUG_TYPE, "ForceStop", &CB)
             << "Won't attempt inlining because module size grew too much.";
    });
    return std::make_unique<InlineAdvice>(this, CB, ORE, Mandatory);
  }

  int CostEstimate = 0;
  if (!Mandatory) {
    auto IsCallSiteInlinable =
        llvm::getInliningCostEstimate(CB, TIR, GetAssumptionCache);
    if (!IsCallSiteInlinable) {
      // We can't inline this for correctness reasons, so return the base
      // InlineAdvice, as we don't care about tracking any state changes (which
      // won't happen).
      return std::make_unique<InlineAdvice>(this, CB, ORE, false);
    }
    CostEstimate = *IsCallSiteInlinable;
  }

  if (Mandatory)
    return getMandatoryAdvice(CB, ORE);

  auto NrCtantParams = 0;
  for (auto I = CB.arg_begin(), E = CB.arg_end(); I != E; ++I) {
    NrCtantParams += (isa<Constant>(*I));
  }

  auto &CallerBefore = FAM.getResult<FunctionPropertiesAnalysis>(Caller);
  auto &CalleeBefore = FAM.getResult<FunctionPropertiesAnalysis>(Callee);

  ModelRunner->setFeature((size_t)InlineFeatureIndex::CalleeBasicBlockCount,
                          CalleeBefore.BasicBlockCount);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CallSiteHeight,
                          FunctionLevels[&Caller]);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::NodeCount, NodeCount);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::NrCtantParams, NrCtantParams);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CostEstimate, CostEstimate);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::EdgeCount, EdgeCount);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CallerUsers, CallerBefore.Uses);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CallerConditionallyExecutedBlocks,
                          CallerBefore.BlocksReachedFromConditionalInstruction);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CallerBasicBlockCount,
                          CallerBefore.BasicBlockCount);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CalleeConditionallyExecutedBlocks,
                          CalleeBefore.BlocksReachedFromConditionalInstruction);
  ModelRunner->setFeature((size_t)InlineFeatureIndex::CalleeUsers, CalleeBefore.Uses);
  return getAdviceFromModel(CB, ORE);
}

std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getAdviceFromModel(CallBase &CB,
                                    OptimizationRemarkEmitter &ORE) {
  return std::make_unique<MLInlineAdvice>(this, CB, ORE, ModelRunner->run());
}

std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getMandatoryAdvice(CallBase &CB,
                                    OptimizationRemarkEmitter &ORE) {
  return std::make_unique<MLInlineAdvice>(this, CB, ORE, true);
}

void MLInlineAdvice::reportContextForRemark(
    DiagnosticInfoOptimizationBase &OR) {
  using namespace ore;
  OR << NV("Callee", Callee->getName());
  for (size_t I = 0; I < NumberOfFeatures; ++I)
    OR << NV(FeatureNameMap[I], getAdvisor()->getModelRunner().getFeature(I));
  OR << NV("ShouldInline", isInliningRecommended());
}

void MLInlineAdvice::recordInliningImpl() {
  ORE.emit([&]() {
    OptimizationRemark R(DEBUG_TYPE, "InliningSuccess", DLoc, Block);
    reportContextForRemark(R);
    return R;
  });
  getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ false);
}

void MLInlineAdvice::recordInliningWithCalleeDeletedImpl() {
  ORE.emit([&]() {
    OptimizationRemark R(DEBUG_TYPE, "InliningSuccessWithCalleeDeleted", DLoc,
                         Block);
    reportContextForRemark(R);
    return R;
  });
  getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ true);
}

void MLInlineAdvice::recordUnsuccessfulInliningImpl(
    const InlineResult &Result) {
  ORE.emit([&]() {
    OptimizationRemarkMissed R(DEBUG_TYPE, "InliningAttemptedAndUnsuccessful",
                               DLoc, Block);
    reportContextForRemark(R);
    return R;
  });
}
void MLInlineAdvice::recordUnattemptedInliningImpl() {
  ORE.emit([&]() {
    OptimizationRemarkMissed R(DEBUG_TYPE, "IniningNotAttempted", DLoc, Block);
    reportContextForRemark(R);
    return R;
  });
}
#endif // defined(LLVM_HAVE_TF_AOT) || defined(LLVM_HAVE_TF_API)
