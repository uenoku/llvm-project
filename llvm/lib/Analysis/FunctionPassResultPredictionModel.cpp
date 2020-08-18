#include "llvm/Analysis/FunctionPassResultPredictionModel.h"

#include "BDCEModel.h"
#include "ECSEModel.h"
#include "InstSimplifyModel.h"
#include "JumpThreadingModel.h"
#include "LCSSAModel.h"
#include "LoopSimplifyModel.h"
#include "LoopUnrollModel.h"
#include "SLPVecModel.h"
#include "SROAModel.h"
#include "SimplifyCFGModel.h"
#include "TailCallEliminationModel.h"

#include "llvm/Support/Debug.h"

namespace llvm {
PassResultPredictionModelRunner::PassResultPredictionModelRunner(
    LLVMContext &Ctx)
    : MLModelRunner(Ctx) {
  // assert(CompiledModel && "The CompiledModel should be valid");

  // FeatureIndices.reserve(NumberOfPassResultPredictionFeatures);

  // for (size_t I = 0; I < NumberOfPassResultPredictionFeatures; ++I) {
  //   const int Index =
  //       CompiledModel->LookupArgIndex(FeatureNameMap[I]);
  //   assert(Index >= 0 && "Cannot find Feature in inlining model");
  //   FeatureIndices[I] = Index;
  // }

  // //  ResultIndex =
  // //  CompiledModel->LookupResultIndex(std::string(FetchPrefix) +
  // DecisionName); assert(ResultIndex >= 0 && "Cannot find DecisionName in
  // inlining model");
}

// template <class T>
// int64_t PassResultPredictionModelRunner::getFeature(int Index) const {
//   //   return *static_cast<int64_t *>(
//   //       CompiledModel->arg_data(FeatureIndices[Index]));
//   return 0;
// }

// void PassResultPredictionModelRunner::setFeature(size_t Index, int64_t Value)
// {
//   //   *static_cast<int64_t *>(CompiledModel->arg_data(
//   //       FeatureIndices[static_cast<size_t>(Index)])) = Value;
// }

// bool PassResultPredictionModelRunner::run() {
//   //   CompiledModel->Run();
//   //   return static_cast<bool>(
//   //       *static_cast<int64_t *>(CompiledModel->result_data(ResultIndex)));
//   return true;
// }
PassResultPredictionModelRunner *
FunctionPassResultPredictionModel::get(std::string name,
                                       llvm::LLVMContext &ctx) {
#define REGISTER(NAME, MODELNAME)                                              \
  {                                                                            \
    if (name == #NAME)                                                         \
      Models[name] = std::make_unique<ModelImpl<MODELNAME>>(ctx);              \
  }

  // dbgs() << "[FunctionPassResultPredictionModel]:get Model " << name << "\n";
  if (Models.find(name) != Models.end())
    return Models[name].get();

  REGISTER(SROA, SROAModel);
  REGISTER(LoopUnrollPass, LoopUnrollModel);
//   REGISTER(LoopSimplifyPass, LoopSimplifyModel);
  REGISTER(BDCEPass, BDCEModel);
  REGISTER(EarlyCSEPass, ECSEModel);
  REGISTER(JumpThreadingPass, JumpThreadingModel);
  REGISTER(TailCallElimPass, TailCallEliminationModel);
  REGISTER(SLPVectorizerPass, SLPVecModel);
  // REGISTER(LCSSAPass, LCSSAModel);
  REGISTER(SimplifyCFGPass, SimplifyCFGModel);
  REGISTER(InstSimplifyPass, InstSimplifyModel);

#undef REGISTER

  return Models[name].get();
}

} // namespace llvm
