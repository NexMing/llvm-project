// RUN: fir-opt %s --cg-rewrite --fir-to-scf --fir-to-mlir --cse --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @_QPmatadd_fixed(
// CHECK-SAME:      %[[ARG0:.*]]: memref<f32> {fir.bindc_name = "a"},
// CHECK-SAME:      %[[ARG1:.*]]: memref<f32> {fir.bindc_name = "b"},
// CHECK-SAME:      %[[ARG2:.*]]: memref<f32> {fir.bindc_name = "c"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 200 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() {bindc_name = "i", in_type = i32, uniq_name = "_QFmatadd_fixedEi"} : memref<i32>
// CHECK:           %[[ALLOCA_1:.*]] = memref.alloca() {bindc_name = "j", in_type = i32, uniq_name = "_QFmatadd_fixedEj"} : memref<i32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_2]]) -> (i32) {
// CHECK:             memref.store %[[VAL_1]], %[[ALLOCA_0]][] : memref<i32>
// CHECK:             %[[FOR_1:.*]] = scf.for %[[VAL_2:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_0]] step %[[CONSTANT_4]] iter_args(%[[VAL_3:.*]] = %[[CONSTANT_2]]) -> (i32) {
// CHECK:               memref.store %[[VAL_3]], %[[ALLOCA_1]][] : memref<i32>
// CHECK:               %[[LOAD_0:.*]] = memref.load %[[ALLOCA_0]][] : memref<i32>
// CHECK:               %[[LOAD_1:.*]] = memref.load %[[ALLOCA_1]][] : memref<i32>
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [100, 200], strides: [1, 100] : memref<f32> to memref<100x200xf32, strided<[1, 100]>>
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[LOAD_0]] : i32 to index
// CHECK:               %[[SUBI_0:.*]] = arith.subi %[[INDEX_CAST_0]], %[[CONSTANT_4]] : index
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[LOAD_1]] : i32 to index
// CHECK:               %[[SUBI_1:.*]] = arith.subi %[[INDEX_CAST_1]], %[[CONSTANT_4]] : index
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[REINTERPRET_CAST_0]]{{\[}}%[[SUBI_0]], %[[SUBI_1]]] [1, 1] [1, 1] : memref<100x200xf32, strided<[1, 100]>> to memref<f32, strided<[], offset: ?>>
// CHECK:               %[[LOAD_2:.*]] = memref.load %[[SUBVIEW_0]][] : memref<f32, strided<[], offset: ?>>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [100, 200], strides: [1, 100] : memref<f32> to memref<100x200xf32, strided<[1, 100]>>
// CHECK:               %[[SUBVIEW_1:.*]] = memref.subview %[[REINTERPRET_CAST_1]]{{\[}}%[[SUBI_0]], %[[SUBI_1]]] [1, 1] [1, 1] : memref<100x200xf32, strided<[1, 100]>> to memref<f32, strided<[], offset: ?>>
// CHECK:               %[[LOAD_3:.*]] = memref.load %[[SUBVIEW_1]][] : memref<f32, strided<[], offset: ?>>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_2]], %[[LOAD_3]] fastmath<contract> : f32
// CHECK:               %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ARG2]] to offset: [0], sizes: [100, 200], strides: [1, 100] : memref<f32> to memref<100x200xf32, strided<[1, 100]>>
// CHECK:               %[[SUBVIEW_2:.*]] = memref.subview %[[REINTERPRET_CAST_2]]{{\[}}%[[SUBI_0]], %[[SUBI_1]]] [1, 1] [1, 1] : memref<100x200xf32, strided<[1, 100]>> to memref<f32, strided<[], offset: ?>>
// CHECK:               memref.store %[[ADDF_0]], %[[SUBVIEW_2]][] : memref<f32, strided<[], offset: ?>>
// CHECK:               %[[LOAD_4:.*]] = memref.load %[[ALLOCA_1]][] : memref<i32>
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONSTANT_2]] overflow<nsw> : i32
// CHECK:               scf.yield %[[ADDI_0]] : i32
// CHECK:             } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>}
// CHECK:             memref.store %[[FOR_1]], %[[ALLOCA_1]][] : memref<i32>
// CHECK:             %[[LOAD_5:.*]] = memref.load %[[ALLOCA_0]][] : memref<i32>
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[LOAD_5]], %[[CONSTANT_2]] overflow<nsw> : i32
// CHECK:             scf.yield %[[ADDI_1]] : i32
// CHECK:           } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>}
// CHECK:           memref.store %[[FOR_0]], %[[ALLOCA_0]][] : memref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPmatadd_fixed(%arg0: !fir.ref<!fir.array<100x200xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<100x200xf32>> {fir.bindc_name = "b"}, %arg2: !fir.ref<!fir.array<100x200xf32>> {fir.bindc_name = "c"}) {
    %c1 = arith.constant 1 : index
    %c200 = arith.constant 200 : index
    %c100 = arith.constant 100 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.shape %c100, %c200 : (index, index) -> !fir.shape<2>
    %2 = fir.declare %arg0(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFmatadd_fixedEa"} : (!fir.ref<!fir.array<100x200xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<100x200xf32>>
    %3 = fir.declare %arg1(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFmatadd_fixedEb"} : (!fir.ref<!fir.array<100x200xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<100x200xf32>>
    %4 = fir.declare %arg2(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFmatadd_fixedEc"} : (!fir.ref<!fir.array<100x200xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<100x200xf32>>
    %5 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmatadd_fixedEi"}
    %6 = fir.declare %5 {uniq_name = "_QFmatadd_fixedEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFmatadd_fixedEj"}
    %8 = fir.declare %7 {uniq_name = "_QFmatadd_fixedEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg3 = %c1 to %c100 step %c1 iter_args(%arg4 = %9) -> (i32) {
        fir.store %arg4 to %6 : !fir.ref<i32>
        %11 = fir.do_loop %arg5 = %c1 to %c200 step %c1 iter_args(%arg6 = %9) -> (i32) {
            fir.store %arg6 to %8 : !fir.ref<i32>
            %14 = fir.load %6 : !fir.ref<i32>
            %15 = fir.convert %14 : (i32) -> i64
            %16 = fir.load %8 : !fir.ref<i32>
            %17 = fir.convert %16 : (i32) -> i64
            %18 = fir.array_coor %2(%1) %15, %17 : (!fir.ref<!fir.array<100x200xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
            %19 = fir.load %18 : !fir.ref<f32>
            %20 = fir.array_coor %3(%1) %15, %17 : (!fir.ref<!fir.array<100x200xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
            %21 = fir.load %20 : !fir.ref<f32>
            %22 = arith.addf %19, %21 fastmath<contract> : f32
            %23 = fir.array_coor %4(%1) %15, %17 : (!fir.ref<!fir.array<100x200xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
            fir.store %22 to %23 : !fir.ref<f32>
            %24 = fir.load %8 : !fir.ref<i32>
            %25 = arith.addi %24, %9 overflow<nsw> : i32
            fir.result %25 : i32
        }
        fir.store %11 to %8 : !fir.ref<i32>
        %12 = fir.load %6 : !fir.ref<i32>
        %13 = arith.addi %12, %9 overflow<nsw> : i32
        fir.result %13 : i32
    }
    fir.store %10 to %6 : !fir.ref<i32>
    return
}
