# The LLVM Compiler Infrastructure
This directory and its sub-directories contain source code for LLVM, a toolkit for the construction of highly optimized compilers, optimizers, and run-time environments.

This branch contains the prototype of pass result prediction framework during GSoC 2020 "Advanced Heuristics for Ordering Compiler Optimization Passes". 
Please refer [the project report](https://docs.google.com/document/d/1pbUPRSjYL5QHLEkwNTjnvdYvgiKaYO_LpyNaTKhTWEA/edit#heading=h.uj16i1ekvivz) for the motivation and detail of the work. 

## usage
### Build
```
$ mkdir build && cd build
$ cmake  ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=On -DLLVM_USE_NEWPM=On -DLLVM_HAVE_TF_AOT=On -DTENSORFLOW_AOT_PATH=~/.local/lib/python3.8/site-packages/tensorflow 
```
Note that without Tensorflow AOT, the build will fail. 

### Run with Inference 
Make sure that you are using New Pass Manager
```
$ ./build/bin/clang++ foo.cpp -O3 -fexperimental-new-pass-manager -mllvm --run-batch-prediction

```

### Dumping Data by using lnt
It's not recommend to run
```
$ lnt runtest test-suite  --sandbox ~/sandbox/ --cc $LLVM_HOME/bulid/bin/clang --cxx $LLVM_HOME/bulid/bin/clang -j16 --test-suite ~/llvm-test-suite/ --cppflags="-fexperimental-new-pass-manager -O3 -mllvm --dump-all-result
$ python3 $LLVM_HOME/script/accmulate_data.py ~/sandbox/test-xxx
```
This creates data.csv to the current directory. 

## Description of Addition
1. I have added/modified following files:
    * `llvm/IR/PassManager.h`
     
       In PassManager, the predictor is called before the pass execcution (only before function passes).  
       https://github.com/uenoku/llvm-project/blob/87b41978c6259553431118d3a1edcf899134685f/llvm/include/llvm/IR/PassManager.h#L542
       ```c++
       PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM,
                        ExtraArgTs... ExtraArgs) {
        ...
        if (predictPassResult(IR, AM, P->name())) 
        // If predictor says "Yes", we run the pass
          PassPA = P->run(IR, AM, ExtraArgs...);

       ```
    * `llvm/Analysis/MLPassResultPredictor.{h, cpp}`
    
       This file defines Pass Result Predictor framework interfaces. `MLPassResultPredcitor<IRUnit,AnalysisManger>` has `predictPassResult` method, which takes pass name (i.e. SROA, GVN...), IR (i.e. Function, Module...) as inputs and returns the estimated result of the pass. In that file, we can define prediction algorithms seprately. Currently, our predictor can predict 11 function passes. If the predictor takes other passes, the predictor simply predict that there is a change.  
    * `llvm/Analysis/PassReulstAnalysis.{h, cpp}`
    
       This defineds Analysis pass whose result is a boolean vector to store predictions from the model. Model definition is `llvm/Analysis/models/all_model/`.
    
