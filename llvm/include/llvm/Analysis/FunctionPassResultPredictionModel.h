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

class PassResultPredictionModelRunner : public MLModelRunner {
public:
  PassResultPredictionModelRunner(LLVMContext &Ctx);
  virtual ~PassResultPredictionModelRunner() = default;

  virtual bool run() = 0;

  virtual void setFeature(size_t Index, int64_t Value) = 0;
  virtual int64_t getFeature(int Index) const = 0;

  //   return 0;
  // }

// private:
//   std::vector<int32_t> FeatureIndices;
//   int32_t ResultIndex = -1;
  //  std::unique_ptr<T> CompiledModel;
  //  T *CompiledModel;
};


struct FunctionPassResultPredictionModel {
  std::map<std::string, std::unique_ptr<PassResultPredictionModelRunner>> Models;
  PassResultPredictionModelRunner* get(std::string, LLVMContext&ctx);
};

template<class T> 
class ModelImpl:  public PassResultPredictionModelRunner {
public:
  ModelImpl(LLVMContext &Ctx): PassResultPredictionModelRunner(Ctx), CompiledModel(std::make_unique<T>()){
    // dbgs()<< "Init\n";
    assert(CompiledModel && "CompiledModel should be valid");
  };
  ~ModelImpl() = default;

  void setFeature(size_t Index, int64_t Value) {
    // dbgs() << "Value " << Value;;
    CompiledModel->arg0(0, Index) = Value;
    assert(CompiledModel->arg0(0, Index) == Value);
  }

  bool run() override {
      CompiledModel->Run();
      auto res = CompiledModel->result0(0, 0);
      // dbgs() << "RawValue" << " " << res << "\n";
      return res > 0.5;
  }
  int64_t getFeature(int Index) const override {
    // temp
    return 0;

  }
//  int64_t getFeature(int Index) const override;
  std::unique_ptr<T> CompiledModel;
};
} // namespace llvm

#endif