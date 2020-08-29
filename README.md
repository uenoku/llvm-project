# The LLVM Compiler Infrastructure
This branch contains prototype of pass result prediction framework during GSoC 2020 "Advanced Heuristics for Ordering Compiler Optimization Passes". 
Please refer [the project report](https://docs.google.com/document/d/1pbUPRSjYL5QHLEkwNTjnvdYvgiKaYO_LpyNaTKhTWEA/edit#heading=h.uj16i1ekvivz) for detail of the work. 


## usage
```
# Build 
$ mkdir build && cd build
$ cmake  ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=On -DLLVM_USE_NEWPM=On -DLLVM_HAVE_TF_AOT=On -DTENSORFLOW_AOT_PATH=~/.local/lib/python3.8/site-packages/tensorflow 
# Note taht without Tensorflow AOT, the build will fail. 

# Run Inference (!!Make sure that you are using New Pass Manager)
$ ./build/bin/clang++ foo.cpp -O3 -mllvm --run-prediction
# ./build/bin/clang++ foo.cpp -O3 -mllvm --run-prediction -mllvm
```

## Description of Addition
1. I have added following files:
    * llvm/Analysis/MLPassResultPredictor.{h, cpp}
       This file defines Pass Result Predictor framework interfaces. `MLPassResultPredcitor<IRUnit,AnalysisManger>` has `predictPassResult` method, which takes pass name (i.e. SROA, GVN...), IR (i.e. Function, Module...) as inputs and returns the estimated result of the pass. In that file, we can define prediction algorithms seprately. 
       
