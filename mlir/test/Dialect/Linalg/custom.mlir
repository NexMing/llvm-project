// RUN: mlir-opt %s -split-input-file | FileCheck %s --check-prefix=CHECK
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s --check-prefix=BUFFERIZE

// CHECK-LABEL: func.func @custom_tensor
// CHECK-SAME:    %[[ARG0:[^:]+]]: tensor<4xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:         %[[CUSTOM:.*]] = linalg.custom
// CHECK-SAME:      domain_name = "vendor_npu"
// CHECK-SAME:      implementation_attrs = "opaque"
// CHECK-SAME:      operator_name = "fused_relu"
// CHECK-SAME:    ins(%[[ARG0]], %{{.*}} : tensor<4xf32>, f32)
// CHECK-SAME:    outs(%[[EMPTY]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:         return %[[CUSTOM]]
func.func @custom_tensor(%arg0: tensor<4xf32>, %scale: f32) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.custom {
      domain_name = "vendor_npu",
      implementation_attrs = "opaque",
      operator_name = "fused_relu"}
    ins(%arg0, %scale : tensor<4xf32>, f32)
    outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// BUFFERIZE-LABEL: func.func @custom_tensor
// BUFFERIZE-SAME:    %[[ARG0:[^:]+]]: memref<4xf32
// BUFFERIZE:         %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// BUFFERIZE:         linalg.custom
// BUFFERIZE-SAME:      domain_name = "vendor_npu"
// BUFFERIZE-SAME:      implementation_attrs = "opaque"
// BUFFERIZE-SAME:      operator_name = "fused_relu"
// BUFFERIZE-SAME:    ins(%[[ARG0]], %{{.*}} : memref<4xf32{{.*}}, f32)
// BUFFERIZE-SAME:    outs(%[[ALLOC]] : memref<4xf32>)

// -----

// CHECK-LABEL: func.func @custom_memref
// CHECK-SAME:    %[[ARG0:[^:]+]]: memref<4xf32>
// CHECK-SAME:    %[[ARG1:[^:]+]]: memref<4xf32>
// CHECK:         linalg.custom
// CHECK-SAME:      domain_name = "vendor_npu"
// CHECK-SAME:      implementation_attrs = "opaque"
// CHECK-SAME:      operator_name = "fused_relu"
// CHECK-SAME:    ins(%[[ARG0]] : memref<4xf32>) outs(%[[ARG1]] : memref<4xf32>)
func.func @custom_memref(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
  linalg.custom {
      domain_name = "vendor_npu",
      implementation_attrs = "opaque",
      operator_name = "fused_relu"}
    ins(%arg0 : memref<4xf32>)
    outs(%arg1 : memref<4xf32>)
  return
}
