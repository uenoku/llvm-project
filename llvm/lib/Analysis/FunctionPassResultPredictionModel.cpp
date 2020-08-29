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
PassAllResultPredictionModelRunner::PassAllResultPredictionModelRunner(
    LLVMContext &Ctx)
    : MLModelRunner(Ctx) {}
PassResultPredictionModelRunner::PassResultPredictionModelRunner(
    LLVMContext &Ctx)
    : MLModelRunner(Ctx) {}

PassResultPredictionModelRunner *
FunctionPassResultPredictionModel::get(std::string name,
                                       llvm::LLVMContext &ctx) {
#define REGISTER(NAME, MODELNAME)                                              \
  {                                                                            \
    if (name == #NAME)                                                         \
      Models[name] = std::make_shared<ModelImpl<MODELNAME>>(ctx);              \
  }

  if (Models.find(name) != Models.end())
    return Models[name].get();

  REGISTER(SROA, SROAModel);
  REGISTER(LoopUnrollPass, LoopUnrollModel);
  REGISTER(BDCEPass, BDCEModel);
  REGISTER(EarlyCSEPass, ECSEModel);
  REGISTER(JumpThreadingPass, JumpThreadingModel);
  REGISTER(TailCallElimPass, TailCallEliminationModel);
  REGISTER(SLPVectorizerPass, SLPVecModel);
  REGISTER(SimplifyCFGPass, SimplifyCFGModel);
  REGISTER(InstSimplifyPass, InstSimplifyModel);
#undef REGISTER

  return Models[name].get();
}

} // namespace llvm