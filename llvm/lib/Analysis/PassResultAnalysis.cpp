#include "llvm/Analysis/PassResultAnalysis.h"
#include "AllPredictModel.h"
#include "llvm/Analysis/FunctionPassResultPredictionModel.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "pass-result-ana"

STATISTIC(NumActualInference, "Number of actitual run");

using namespace llvm;

namespace llvm {
class AllPassResultPredictionModelRunner
    : public PassAllResultPredictionModelRunner {
public:
  AllPassResultPredictionModelRunner(LLVMContext &Ctx)
      : PassAllResultPredictionModelRunner(Ctx),
        CompiledModel(std::make_unique<AllPredictModel>()){};
  ~AllPassResultPredictionModelRunner() = default;
  bool calc = false;
  bool run() override {
    calc = true;
    LLVM_DEBUG({ dbgs() << "running\n"; });
    return CompiledModel->Run();
  }
  bool get_result(size_t Index) {
    if (!calc)
      run();
    auto res = (CompiledModel->result0(0, Index) > 0.0);
    return res;
  }
  float *get_result_ptr() {
    if (!calc)
      run();
    return CompiledModel->result0_data();
  }
  void setFeature(size_t Index, int64_t Value) override {
    calc = false;
    LLVM_DEBUG(dbgs() << Index << " " << Value << "\n";);
    CompiledModel->arg0(0, Index) = Value;
    assert(CompiledModel->arg0(0, Index) == Value);
  }
  int64_t getFeature(int Index) const override {
    // temp
    return 0;
  }

  std::unique_ptr<AllPredictModel> CompiledModel;
};
FunctionPassResultPrediction
FunctionPassResultPrediction::getFunctionResultPrediction(
    Function &F, FunctionAnalysisManager &FAM) {

  // auto prev = FAM.getCachedResult<FunctionPassResultAnalysis>(F);
  // if (prev) {
  //   prev->counter++;
  //   if (prev->counter % PredictionReuse != 0) {
  //     return *prev;
  //   }
  // }

  {
    auto pred = F.getContext().getPredictor();
    if (!pred) {
      auto &Ctx = F.getContext();
      FunctionPassResultPredictionModel FPe;
      Ctx.setPredictor(std::make_unique<FunctionPassResultPredictionModel>());
      Ctx.getPredictor()->all_pred =
          std::make_shared<AllPassResultPredictionModelRunner>(F.getContext());
    }
  }

  auto pred = F.getContext().getPredictor()->all_pred;
  assert(pred);
  auto FPAna = FAM.getResult<FunctionPropertiesAnalysis>(F);
  auto FP = FPAna.toVec();
  assert(FP.size() == 87);
  LLVM_DEBUG(dbgs() << "set features\n");
  for (int i = 0; i < FP.size(); i++) {
    pred->setFeature(i, FP[i]);
  }
  LLVM_DEBUG(dbgs() << "done set features\n");

  FunctionPassResultPrediction frp;

  auto t = pred->get_result_ptr();
  for (int i = 0; i < 11; i++) {
    frp.result.push_back(t[i] > 0.5);
  }
  LLVM_DEBUG({
    for (int i = 0; i < 11; i++) {
      dbgs() << i << " " << frp.result[i] << "\n";
    }
  };);

  NumActualInference++;
  return frp;
}

FunctionPassResultAnalysis::Result
FunctionPassResultAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  return FunctionPassResultPrediction::getFunctionResultPrediction(F, FAM);
}

AnalysisKey FunctionPassResultAnalysis::Key;

} // namespace llvm