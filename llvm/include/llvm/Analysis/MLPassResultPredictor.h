//===- MLInlineAdvisor.h - ML - based InlineAdvisor factories ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MLPASSRESULT_PREDICTOR_H
#define LLVM_ANALYSIS_MLPASSRESULT_PREDICTOR_H
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/TypeName.h"
#include <memory>

namespace llvm {
class Function;
class Module;
class MLInlineAdvice;
struct PredictorInput;
/// ML guided Pass Result Predictor
template <typename IRUnitT, typename AnalysisManagerT>
struct MLPassResultPredictor {
public:
  MLPassResultPredictor() {}
  virtual ~MLPassResultPredictor() = default;
  bool predict(PredictorInput *In, LLVMContext& Ctx);
  PredictorInput *createInput(IRUnitT &IR, AnalysisManagerT &AM,
                              StringRef PassName);
  void dump(StringRef, PredictorInput *, bool, raw_ostream &);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MLPASSRESULT_PREDICTOR_H
