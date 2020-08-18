if (DEFINED LLVM_HAVE_TF_AOT OR DEFINED LLVM_HAVE_TF_API)
  if (DEFINED LLVM_HAVE_TF_AOT)
    include(TensorFlowCompile)

    tfcompile(models/inliner serve action InlinerSizeModel llvm::InlinerSizeModel)
    #if (DEFINED LLVM_LARGE_MODEL)
    #  tfcompile(models/models_2424/sroa serve predict SROAModel llvm::SROAModel)
    #  tfcompile(models/models_2424/bdce serve predict BDCEModel llvm::BDCEModel)
    #  tfcompile(models/models_2424/ecse serve predict ECSEModel llvm::ECSEModel)
    #  tfcompile(models/models_2424/inst_simplify serve predict InstSimplifyModel llvm::InstSimplifyModel)
    #  tfcompile(models/models_2424/jump_threading serve predict JumpThreadingModel llvm::JumpThreadingModel)
    #  tfcompile(models/models_2424/lcssa serve predict LCSSAModel llvm::LCSSAModel)
    #  tfcompile(models/models_2424/lsimplify serve predict LoopSimplifyModel llvm::LoopSimplifyModel)
    #  tfcompile(models/models_2424/scfg serve predict SimplifyCFGModel llvm::SimplifyCFGModel)
    #  tfcompile(models/models_2424/slp_vec serve predict SLPVecModel llvm::SLPVecModel)
    #  tfcompile(models/models_2424/unroll serve predict LoopUnrollModel llvm::LoopUnrollModel)
    #  tfcompile(models/models_2424/tce serve predict TailCallEliminationModel llvm::TailCallEliminationModel)
    #endif()
    #if (DEFINED LLVM_SMALL_MODEL)
    #  tfcompile(models/models_small/sroa serve predict SROAModel llvm::SROAModel)
    #  tfcompile(models/models_small/bdce serve predict BDCEModel llvm::BDCEModel)
    #  tfcompile(models/models_small/ecse serve predict ECSEModel llvm::ECSEModel)
    #  tfcompile(models/models_small/inst_simplify serve predict InstSimplifyModel llvm::InstSimplifyModel)
    #  tfcompile(models/models_small/jump_threading serve predict JumpThreadingModel llvm::JumpThreadingModel)
    #  tfcompile(models/models_small/lcssa serve predict LCSSAModel llvm::LCSSAModel)
    #  tfcompile(models/models_small/lsimplify serve predict LoopSimplifyModel llvm::LoopSimplifyModel)
    #  tfcompile(models/models_small/scfg serve predict SimplifyCFGModel llvm::SimplifyCFGModel)
    #  tfcompile(models/models_small/slp_vec serve predict SLPVecModel llvm::SLPVecModel)
    #  tfcompile(models/models_small/unroll serve predict LoopUnrollModel llvm::LoopUnrollModel)
    #  tfcompile(models/models_small/tce serve predict TailCallEliminationModel llvm::TailCallEliminationModel)
    #  tfcompile(models/models_small/gvn serve predict GVNModel llvm::GVNModel)
    #endif()
    if (DEFINED LLVM_MIN_MODEL)
      tfcompile(models/models_min/sroa serve predict SROAModel llvm::SROAModel)
      tfcompile(models/models_min/bdce serve predict BDCEModel llvm::BDCEModel)
      tfcompile(models/models_min/ecse serve predict ECSEModel llvm::ECSEModel)
      tfcompile(models/models_min/inst_simplify serve predict InstSimplifyModel llvm::InstSimplifyModel)
      tfcompile(models/models_min/jump_threading serve predict JumpThreadingModel llvm::JumpThreadingModel)
      tfcompile(models/models_min/lcssa serve predict LCSSAModel llvm::LCSSAModel)
      tfcompile(models/models_min/lsimplify serve predict LoopSimplifyModel llvm::LoopSimplifyModel)
      tfcompile(models/models_min/scfg serve predict SimplifyCFGModel llvm::SimplifyCFGModel)
      tfcompile(models/models_min/slp_vec serve predict SLPVecModel llvm::SLPVecModel)
      tfcompile(models/models_min/unroll serve predict LoopUnrollModel llvm::LoopUnrollModel)
      tfcompile(models/models_min/tce serve predict TailCallEliminationModel llvm::TailCallEliminationModel)
      tfcompile(models/models_min/gvn serve predict GVNModel llvm::GVNModel)
    endif()

    list(APPEND GeneratedMLSources
      $<TARGET_OBJECTS:tf_xla_runtime_objects>
      ${GENERATED_OBJS}
    )
  endif()

  if (DEFINED LLVM_HAVE_TF_API)
    LIST(APPEND MLLinkDeps ${tensorflow_c_api})
  endif()
endif()

add_llvm_component_library(LLVMAnalysis
  AliasAnalysis.cpp
  AliasAnalysisEvaluator.cpp
  AliasAnalysisSummary.cpp
  AliasSetTracker.cpp
  Analysis.cpp
  AssumeBundleQueries.cpp
  AssumptionCache.cpp
  BasicAliasAnalysis.cpp
  BlockFrequencyInfo.cpp
  BlockFrequencyInfoImpl.cpp
  BranchProbabilityInfo.cpp
  CFG.cpp
  CFGPrinter.cpp
  CFLAndersAliasAnalysis.cpp
  CFLSteensAliasAnalysis.cpp
  CGSCCPassManager.cpp
  CallGraph.cpp
  CallGraphSCCPass.cpp
  CallPrinter.cpp
  CaptureTracking.cpp
  CmpInstAnalysis.cpp
  CostModel.cpp
  CodeMetrics.cpp
  ConstantFolding.cpp
  DDG.cpp
  Delinearization.cpp
  DemandedBits.cpp
  DependenceAnalysis.cpp
  DependenceGraphBuilder.cpp
  DevelopmentModeInlineAdvisor.cpp
  DivergenceAnalysis.cpp
  DomPrinter.cpp
  DomTreeUpdater.cpp
  DominanceFrontier.cpp
  EHPersonalities.cpp
  FunctionPropertiesAnalysis.cpp
  FunctionPassResultPredictionModel.cpp
  GlobalsModRef.cpp
  GuardUtils.cpp
  HeatUtils.cpp
  IVDescriptors.cpp
  IVUsers.cpp
  IndirectCallPromotionAnalysis.cpp
  InlineCost.cpp
  InlineAdvisor.cpp
  InlineSizeEstimatorAnalysis.cpp
  InstCount.cpp
  InstructionPrecedenceTracking.cpp
  InstructionSimplify.cpp
  Interval.cpp
  IntervalPartition.cpp
  LazyBranchProbabilityInfo.cpp
  LazyBlockFrequencyInfo.cpp
  LazyCallGraph.cpp
  LazyValueInfo.cpp
  LegacyDivergenceAnalysis.cpp
  Lint.cpp
  Loads.cpp
  LoopAccessAnalysis.cpp
  LoopAnalysisManager.cpp
  LoopCacheAnalysis.cpp
  LoopNestAnalysis.cpp
  LoopUnrollAnalyzer.cpp
  LoopInfo.cpp
  LoopPass.cpp
  MLInlineAdvisor.cpp
  MLPassResultPredictor.cpp
  MemDepPrinter.cpp
  MemDerefPrinter.cpp
  MemoryBuiltins.cpp
  MemoryDependenceAnalysis.cpp
  MemoryLocation.cpp
  MemorySSA.cpp
  MemorySSAUpdater.cpp
  ModuleDebugInfoPrinter.cpp
  ModuleSummaryAnalysis.cpp
  MustExecute.cpp
  ObjCARCAliasAnalysis.cpp
  ObjCARCAnalysisUtils.cpp
  ObjCARCInstKind.cpp
  OptimizationRemarkEmitter.cpp
  PHITransAddr.cpp
  PhiValues.cpp
  PostDominators.cpp
  ProfileSummaryInfo.cpp
  PtrUseVisitor.cpp
  RegionInfo.cpp
  RegionPass.cpp
  RegionPrinter.cpp
  ReleaseModeModelRunner.cpp
  ScalarEvolution.cpp
  ScalarEvolutionAliasAnalysis.cpp
  ScalarEvolutionDivision.cpp
  ScalarEvolutionNormalization.cpp
  StackLifetime.cpp
  StackSafetyAnalysis.cpp
  SyncDependenceAnalysis.cpp
  SyntheticCountsUtils.cpp
  TFUtils.cpp
  TargetLibraryInfo.cpp
  TargetTransformInfo.cpp
  Trace.cpp
  TypeBasedAliasAnalysis.cpp
  TypeMetadataUtils.cpp
  ScopedNoAliasAA.cpp
  ValueLattice.cpp
  ValueLatticeUtils.cpp
  ValueTracking.cpp
  VectorUtils.cpp
  VFABIDemangling.cpp
  ${GeneratedMLSources}

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Analysis

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  ${MLLinkDeps}
  )
