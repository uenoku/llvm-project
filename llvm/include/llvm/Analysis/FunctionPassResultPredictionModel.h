#ifndef LLVM_ANALYSIS_FUNCTION_PASS_RESULT_PREDICTION_MODEL_H
#define LLVM_ANALYSIS_FUNCTION_PASS_RESULT_PREDICTION_MODEL_H
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <memory>

#include <string>

namespace llvm {

class PassResultPredictionModelRunner : public MLModelRunner {
public:
  PassResultPredictionModelRunner(LLVMContext &Ctx);
  virtual ~PassResultPredictionModelRunner() = default;

  virtual bool run() = 0;

  virtual void setFeature(size_t Index, int64_t Value) = 0;
  virtual int64_t getFeature(int Index) const = 0;
};

class PassAllResultPredictionModelRunner : public MLModelRunner {
public:
  PassAllResultPredictionModelRunner(LLVMContext &Ctx);
  virtual ~PassAllResultPredictionModelRunner() = default;

  virtual bool run() = 0;

  virtual void setFeature(size_t Index, int64_t Value) = 0;
  virtual int64_t getFeature(int Index) const = 0;
  virtual std::vector<bool> getResult() = 0;
};

struct FunctionPassResultPredictionModel {
  std::map<std::string, std::shared_ptr<PassResultPredictionModelRunner>>
      Models;

  PassResultPredictionModelRunner *get(std::string, LLVMContext &ctx);
  std::shared_ptr<PassAllResultPredictionModelRunner> ModelRunner;
};

template <class T> class ModelImpl : public PassResultPredictionModelRunner {
public:
  ModelImpl(LLVMContext &Ctx)
      : PassResultPredictionModelRunner(Ctx),
        CompiledModel(std::make_unique<T>()) {
    assert(CompiledModel && "CompiledModel should be valid");
  };
  ~ModelImpl() = default;

  void setFeature(size_t Index, int64_t Value) override {
    CompiledModel->arg0(0, Index) = Value;
    assert(CompiledModel->arg0(0, Index) == Value);
  }

  bool run() override {
    CompiledModel->Run();
    auto Ret = CompiledModel->result0(0, 0);
    DEBUG_WITH_TYPE("raw", dbgs() << "RawValue"
                                  << " " << Ret << "\n");
    return Ret > 0.5;
  }
  int64_t getFeature(int Index) const override {
    return CompiledModel->arg0(0, Index);
  }
  std::unique_ptr<T> CompiledModel;
};
} // namespace llvm

#endif
