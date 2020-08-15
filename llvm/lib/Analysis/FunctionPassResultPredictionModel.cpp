#include "llvm/Analysis/FunctionPassResultPredictionModel.h"

namespace llvm {
  PassResultPredictionModelRunner::PassResultPredictionModelRunner(LLVMContext &Ctx)
      : MLModelRunner(Ctx)
  //     CompiledModel(std::make_unique<llvm::InlinerSizeModel>()) 
  {
  //  assert(CompiledModel && "The CompiledModel should be valid");

    FeatureIndices.reserve(NumberOfPassResultPredictionFeatures);

    for (size_t I = 0; I < NumberOfPassResultPredictionFeatures; ++I) {
      // const int Index =
      //     CompiledModel->LookupArgIndex(FeatureNameMap[I]);
      // assert(Index >= 0 && "Cannot find Feature in inlining model");
      // FeatureIndices[I] = Index;
    }

      //  ResultIndex =
      //  CompiledModel->LookupResultIndex(std::string(FetchPrefix) + DecisionName);
    assert(ResultIndex >= 0 && "Cannot find DecisionName in inlining model");
  }

  int64_t PassResultPredictionModelRunner::getFeature(int Index) const {
  //   return *static_cast<int64_t *>(
  //       CompiledModel->arg_data(FeatureIndices[Index]));
  return 0;
  }

  void PassResultPredictionModelRunner::setFeature(size_t Index, int64_t Value) {
  //   *static_cast<int64_t *>(CompiledModel->arg_data(
  //       FeatureIndices[static_cast<size_t>(Index)])) = Value;
  }

  bool PassResultPredictionModelRunner::run() {
  //   CompiledModel->Run();
  //   return static_cast<bool>(
  //       *static_cast<int64_t *>(CompiledModel->result_data(ResultIndex)));
  return true;
  }
}
