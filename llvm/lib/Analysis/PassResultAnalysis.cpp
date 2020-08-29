#include "llvm/Analysis/FunctionPassResultPredictionModel.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/PassResultAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "pass-result-ana"

STATISTIC(NumActualInference, "Number of actitual run");

using namespace llvm;

#include "AllPredictModel.h"

namespace llvm {
class AllPassResultPredictionModelRunner
    : public PassAllResultPredictionModelRunner {
public:
  AllPassResultPredictionModelRunner(LLVMContext &Ctx)
      : PassAllResultPredictionModelRunner(Ctx),
        CompiledModel(std::make_unique<AllPredictModel>()){};
  ~AllPassResultPredictionModelRunner() = default;
  bool run() override {
    LLVM_DEBUG({ dbgs() << "running\n"; });
    return CompiledModel->Run();
  }

  std::vector<bool> getResult() {
    std::vector<bool> Ret(CompiledModel->result0_count());
    for (int i = 0; i < CompiledModel->result0_count(); i++) {
      Ret[i] = CompiledModel->result0(0, i) > 0.5;
    }
    return Ret;
  }
  void setFeature(size_t Index, int64_t Value) override {
    LLVM_DEBUG(dbgs() << Index << " " << Value << "\n";);
    CompiledModel->arg0(0, Index) = Value;
    assert(CompiledModel->arg0(0, Index) == Value);
  }
  int64_t getFeature(int Index) const override {
    return CompiledModel->arg0(0, Index);
  }

  std::unique_ptr<AllPredictModel> CompiledModel;
};

FunctionPassResultPrediction
FunctionPassResultPrediction::getFunctionResultPrediction(
    Function &F, FunctionAnalysisManager &FAM) {

  {
    auto Predictor = F.getContext().getPredictor();

    // Initialize
    if (!Predictor) {
      auto &Ctx = F.getContext();
      FunctionPassResultPredictionModel FPe;
      Ctx.setPredictor(std::make_unique<FunctionPassResultPredictionModel>());
      Ctx.getPredictor()->ModelRunner =
          std::make_shared<AllPassResultPredictionModelRunner>(F.getContext());
    }
  }

  auto Pred = F.getContext().getPredictor()->ModelRunner;
  auto CodeFeature = FAM.getResult<FunctionPropertiesAnalysis>(F).toVec();
  assert(CodeFeature.size() == 87);
  LLVM_DEBUG(dbgs() << "setting features\n");
  for (int i = 0; i < CodeFeature.size(); i++) {
    Pred->setFeature(i, CodeFeature[i]);
  }
  LLVM_DEBUG(dbgs() << "done set features\n");

  FunctionPassResultPrediction FPRP;
  Pred->run();
  FPRP.result = std::move(Pred->getResult());
  LLVM_DEBUG({
    for (int i = 0; i < FPRP.result.size(); i++) {
      dbgs() << i << " " << FPRP.result[i] << "\n";
    }
  };);
  NumActualInference++;
  return FPRP;
}

FunctionPassResultAnalysis::Result
FunctionPassResultAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  return FunctionPassResultPrediction::getFunctionResultPrediction(F, FAM);
}

AnalysisKey FunctionPassResultAnalysis::Key;

} // namespace llvm