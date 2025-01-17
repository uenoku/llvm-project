; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -instsimplify -S < %s | FileCheck %s

; Most basic case, fully identical PHI nodes
define void @test0(i32 %v0, i32 %v1, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @test0(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V0]], [[B0]] ], [ [[V1]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  ret void
}

; Fully identical PHI nodes, but order of operands differs
define void @test1(i32 %v0, i32 %v1, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V1]], [[B1]] ], [ [[V0]], [[B0]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v1, %b1 ], [ %v0, %b0 ]
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  ret void
}

; Different incoming values in second PHI
define void @negative_test2(i32 %v0, i32 %v1, i32 %v2, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @negative_test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V0]], [[B0]] ], [ [[V2:%.*]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v0, %b0 ], [ %v2, %b1 ] ; from %b0 takes %v2 instead of %v1
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  ret void
}
define void @negative_test3(i32 %v0, i32 %v1, i32 %v2, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @negative_test3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V2:%.*]], [[B1]] ], [ [[V0]], [[B0]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v2, %b1 ], [ %v0, %b0 ] ; from %b0 takes %v2 instead of %v1
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  ret void
}
define void @negative_test4(i32 %v0, i32 %v1, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @negative_test4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V1]], [[B1]] ], [ [[V0]], [[B0]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v1, %b1 ], [ %v0, %b0 ] ; incoming values are swapped
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  ret void
}

; Both PHI's are identical, but the first one has no uses, so ignore it.
define void @test5(i32 %v0, i32 %v1, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ] ; unused
  %i1 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  store i32 %i1, i32* %d1
  ret void
}
; Second PHI has no uses
define void @test6(i32 %v0, i32 %v1, i1 %c, i32* %d0, i32* %d1) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ] ; unused
  store i32 %i0, i32* %d0
  ret void
}

; Non-matching PHI node should be ignored without terminating CSE.
define void @test7(i32 %v0, i32 %v1, i16 %v2, i16 %v3, i1 %c, i32* %d0, i32* %d1, i16* %d2) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[IBAD:%.*]] = phi i16 [ [[V2:%.*]], [[B0]] ], [ [[V3:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V0]], [[B0]] ], [ [[V1]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    store i16 [[IBAD]], i16* [[D2:%.*]], align 2
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %iBAD = phi i16 [ %v2, %b0 ], [ %v3, %b1 ]
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  store i16 %iBAD, i16* %d2
  ret void
}
define void @test8(i32 %v0, i32 %v1, i16 %v2, i16 %v3, i1 %c, i32* %d0, i32* %d1, i16* %d2) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[IBAD:%.*]] = phi i16 [ [[V2:%.*]], [[B0]] ], [ [[V3:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V0]], [[B0]] ], [ [[V1]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    store i16 [[IBAD]], i16* [[D2:%.*]], align 2
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %iBAD = phi i16 [ %v2, %b0 ], [ %v3, %b1 ]
  %i1 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  store i16 %iBAD, i16* %d2
  ret void
}
define void @test9(i32 %v0, i32 %v1, i16 %v2, i16 %v3, i1 %c, i32* %d0, i32* %d1, i16* %d2) {
; CHECK-LABEL: @test9(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C:%.*]], label [[B0:%.*]], label [[B1:%.*]]
; CHECK:       b0:
; CHECK-NEXT:    br label [[END:%.*]]
; CHECK:       b1:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[I0:%.*]] = phi i32 [ [[V0:%.*]], [[B0]] ], [ [[V1:%.*]], [[B1]] ]
; CHECK-NEXT:    [[I1:%.*]] = phi i32 [ [[V0]], [[B0]] ], [ [[V1]], [[B1]] ]
; CHECK-NEXT:    [[IBAD:%.*]] = phi i16 [ [[V2:%.*]], [[B0]] ], [ [[V3:%.*]], [[B1]] ]
; CHECK-NEXT:    store i32 [[I0]], i32* [[D0:%.*]], align 4
; CHECK-NEXT:    store i32 [[I1]], i32* [[D1:%.*]], align 4
; CHECK-NEXT:    store i16 [[IBAD]], i16* [[D2:%.*]], align 2
; CHECK-NEXT:    ret void
;
entry:
  br i1 %c, label %b0, label %b1

b0:
  br label %end

b1:
  br label %end

end:
  %i0 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %i1 = phi i32 [ %v0, %b0 ], [ %v1, %b1 ]
  %iBAD = phi i16 [ %v2, %b0 ], [ %v3, %b1 ]
  store i32 %i0, i32* %d0
  store i32 %i1, i32* %d1
  store i16 %iBAD, i16* %d2
  ret void
}
