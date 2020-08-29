//=- FunctionPropertiesAnalysis.h - Function Properties Analysis --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionPropertiesInfo and FunctionPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUNCTIONRESULTANALYSIS_H_
#define LLVM_FUNCTIONRESULTANALYSIS_H_

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include <map>

namespace llvm {
class Function;

class FunctionPassResultPrediction {
public:
  static FunctionPassResultPrediction
  getFunctionResultPrediction(Function &F, FunctionAnalysisManager &FAM);
  std::vector<bool> result;
  unsigned counter = 0;
};
// Analysis pass
class FunctionPassResultAnalysis
    : public AnalysisInfoMixin<FunctionPassResultAnalysis> {

public:
  static AnalysisKey Key;

  using Result = FunctionPassResultPrediction;

  Result run(Function &F, FunctionAnalysisManager &FAM);
};
} // namespace llvm
#endif // LLVM_FUNCTIONPROPERTIESANALYSIS_H_