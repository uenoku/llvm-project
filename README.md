# The LLVM Compiler Infrastructure
This directory and its sub-directories contain source code for LLVM, a toolkit for the construction of highly optimized compilers, optimizers, and run-time environments.

This branch contains prototype of pass result prediction framework during GSoC 2020 "Advanced Heuristics for Ordering Compiler Optimization Passes". 
Please refer [the project report](https://docs.google.com/document/d/1pbUPRSjYL5QHLEkwNTjnvdYvgiKaYO_LpyNaTKhTWEA/edit#heading=h.uj16i1ekvivz) for the motivation and detail of the work. 

## usage
```
# Build 
$ mkdir build && cd build
$ cmake  ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=On -DLLVM_USE_NEWPM=On -DLLVM_HAVE_TF_AOT=On -DTENSORFLOW_AOT_PATH=~/.local/lib/python3.8/site-packages/tensorflow 
# Note that without Tensorflow AOT, the build will fail. 

# Run Inference (!!Make sure that you are using New Pass Manager)
$ ./build/bin/clang++ foo.cpp -O3 -mllvm --run-prediction

# Dumping Data by using lnt
# lnt runtest test-suite  --sandbox ~/sandbox/ --cc $LLVM_HOME/bulid/bin/clang --cxx $LLVM_HOME/bulid/bin/clang -j16 --test-suite ~/llvm-test-suite/ --cppflags="-fexperimental-new-pass-manager -O3 -mllvm --dump-all-result
# python3 script/accmulate_data.py ~/sandbox/test-xxx
This creates data.csv to the current directory
```

## Description of Addition
1. I have added following files:
    * llvm/Analysis/MLPassResultPredictor.{h, cpp}
       This file defines Pass Result Predictor framework interfaces. `MLPassResultPredcitor<IRUnit,AnalysisManger>` has `predictPassResult` method, which takes pass name (i.e. SROA, GVN...), IR (i.e. Function, Module...) as inputs and returns the estimated result of the pass. In that file, we can define prediction algorithms seprately. In PassManager, the predictor is called before the pass execcution (currently only before function passes).

       
