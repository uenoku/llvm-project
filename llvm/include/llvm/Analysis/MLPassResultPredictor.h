//===- MLInlineAdvisor.h - ML - based InlineAdvisor factories ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MLPASSRESULT_PREDICTOR_H
#define LLVM_ANALYSIS_MLPASSRESULT_PREDICTOR_H
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TypeName.h"
#include <memory>
#include <optional>

namespace llvm {
class Function;
class Module;
/// ML guided Pass Result Predictor
template <typename IRUnitT, typename AnalysisManagerT>
struct MLPassResultPredictor {
public:
  MLPassResultPredictor() {}
  virtual ~MLPassResultPredictor() = default;
  Optional<std::vector<bool>> initializePassResultsPrediction(int LengthOfPassPipeline,
                                                 IRUnitT &IR,
                                                 AnalysisManagerT &AM
                                                 );
  void updatePassResultsPrediction(int LengthOfPassPipeline, IRUnitT &In,
                         AnalysisManagerT &MAM,
                         Optional<std::vector<bool>> &Result, int Index, std::vector<bool> &PreviousResult);
  void dumpAfterPasses(int LengthOfPassPipeline, IRUnitT &In,
                         AnalysisManagerT &MAM,
                         std::vector<bool> &Result);
  bool valid();
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MLPASSRESULT_PREDICTOR_H
