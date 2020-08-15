#ifndef LLVM_ANALYSIS_FUNCTION_PASS_RESULT_PREDICTION_MODEL_H
#define LLVM_ANALYSIS_FUNCTION_PASS_RESULT_PREDICTION_MODEL_H
#include "llvm/Analysis/MLModelRunner.h"
#include <map>
#include <memory>
#include <string>
namespace llvm {

#define PASS_RESULT_FEATURE_ITERATOR(M)                                        \
  M(InstructionCount, "InstructionCount", "InstructionCount")                  \
  M(BasicBlockCount, "BasicBlockCount", "BasicBlockCount")

enum class PassResultPredictionFeatureIndex : size_t {
#define POPULATE_INDICES(INDEX_NAME, NAME, COMMENT) INDEX_NAME,
  PASS_RESULT_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES
      NumberOfFeatures
};

constexpr size_t NumberOfPassResultPredictionFeatures =
    static_cast<size_t>(PassResultPredictionFeatureIndex::NumberOfFeatures);


class PassResultPredictionModelRunner final : public MLModelRunner {
public:
  PassResultPredictionModelRunner(LLVMContext &Ctx);
  virtual ~PassResultPredictionModelRunner() = default;

  bool run() override;

  void setFeature(size_t Index, int64_t Value) override;
  int64_t getFeature(int Index) const override;

private:
  std::vector<int32_t> FeatureIndices;
  int32_t ResultIndex = -1;
//  std::unique_ptr<llvm::InlinerSizeModel> CompiledModel;
};


struct FunctionPassResultPredictionModel {
  std::map<std::string, MLModelRunner*> PassNameToModel;
};
} // namespace llvm

#endif