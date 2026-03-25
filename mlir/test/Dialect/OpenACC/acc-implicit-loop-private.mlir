// RUN: mlir-opt %s -acc-implicit-loop-private -split-input-file | FileCheck %s

// -----

// Test: scalar from acc.firstprivate on enclosing parallel becomes private on loop.
// The loop should have an acc.private clause with varPtr pointing to the
// firstprivate result, initialized from the enclosing parallel's firstprivate.

acc.firstprivate.recipe @firstprivatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
} copy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
}

// CHECK-LABEL: func.func @test_scalar_firstprivate_to_loop_private
// CHECK: [[FP:%.*]] = acc.firstprivate varPtr(%{{.*}} : memref<i32>) recipe(@firstprivatization_memref_i32) -> memref<i32>
// CHECK: acc.parallel firstprivate([[FP]] : memref<i32>) {
// CHECK:   [[LP:%.*]] = acc.private varPtr([[FP]] : memref<i32>) recipe({{@.*}}) -> memref<i32> {implicit = true, name = ""}
// CHECK:   acc.loop private([[LP]] : memref<i32>)
// CHECK:     memref.load [[LP]]
// CHECK-NOT: firstprivate([[FP]]

func.func @test_scalar_firstprivate_to_loop_private(%n : memref<i32>) {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %fp = acc.firstprivate varPtr(%n : memref<i32>)
          recipe(@firstprivatization_memref_i32) -> memref<i32>
          {name = ""}
  acc.parallel firstprivate(%fp : memref<i32>) {
    acc.loop control(%iv : i32) = (%c0 : i32) to (%c10 : i32)
                                  step (%c1 : i32) {
      %val = memref.load %fp[] : memref<i32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// -----

// Test: scalar from acc.private on an outer acc.loop becomes private on the inner loop.
// If the outer loop already has a private clause for the scalar, the inner loop
// should get a new private clause initialized from the outer private result.

acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

// CHECK-LABEL: func.func @test_nested_loop_private
// CHECK: [[OUTER_PRIV:%.*]] = acc.private varPtr(%{{.*}} : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
// CHECK: acc.loop private([[OUTER_PRIV]] : memref<i32>)
// CHECK:   [[INNER_PRIV:%.*]] = acc.private varPtr([[OUTER_PRIV]] : memref<i32>) recipe({{@.*}}) -> memref<i32> {implicit = true, name = ""}
// CHECK:   acc.loop private([[INNER_PRIV]] : memref<i32>)
// CHECK:     memref.load [[INNER_PRIV]]

func.func @test_nested_loop_private(%n : memref<i32>) {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %outerPriv = acc.private varPtr(%n : memref<i32>)
                 recipe(@privatization_memref_i32) -> memref<i32>
                 {name = ""}
  acc.loop private(%outerPriv : memref<i32>)
           control(%i : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
    acc.loop control(%j : i32) = (%c0 : i32) to (%c10 : i32) step (%c1 : i32) {
      %val = memref.load %outerPriv[] : memref<i32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// -----

// Test: aggregate (array) from firstprivate is NOT privatized on the loop —
// only scalars receive implicit private clauses.

acc.firstprivate.recipe @firstprivatization_memref_10xi32 : memref<10xi32> init {
^bb0(%arg0: memref<10xi32>):
  %0 = memref.alloca() : memref<10xi32>
  acc.yield %0 : memref<10xi32>
} copy {
^bb0(%arg0: memref<10xi32>, %arg1: memref<10xi32>):
  acc.terminator
}

// CHECK-LABEL: func.func @test_array_not_privatized
// CHECK: acc.parallel firstprivate(%{{.*}} : memref<10xi32>)
// CHECK: acc.loop control
// CHECK-NOT: acc.loop private
// CHECK-NOT: acc.private

func.func @test_array_not_privatized(%arr : memref<10xi32>) {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %fp = acc.firstprivate varPtr(%arr : memref<10xi32>)
          recipe(@firstprivatization_memref_10xi32) -> memref<10xi32>
          {name = ""}
  acc.parallel firstprivate(%fp : memref<10xi32>) {
    acc.loop control(%iv : i32) = (%c0 : i32) to (%c10 : i32)
                                  step (%c1 : i32) {
      %c0_idx = arith.constant 0 : index
      %val = memref.load %fp[%c0_idx] : memref<10xi32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// -----

// Test: scalar already in loop's private clause is NOT re-added.

acc.firstprivate.recipe @firstprivatization_memref_i64 : memref<i64> init {
^bb0(%arg0: memref<i64>):
  %0 = memref.alloca() : memref<i64>
  acc.yield %0 : memref<i64>
} copy {
^bb0(%arg0: memref<i64>, %arg1: memref<i64>):
  %0 = memref.load %arg0[] : memref<i64>
  memref.store %0, %arg1[] : memref<i64>
  acc.terminator
}

acc.private.recipe @privatization_memref_i64 : memref<i64> init {
^bb0(%arg0: memref<i64>):
  %0 = memref.alloca() : memref<i64>
  acc.yield %0 : memref<i64>
}

// CHECK-LABEL: func.func @test_already_private_not_duplicated
// CHECK: [[FP:%.*]] = acc.firstprivate varPtr
// CHECK: [[LP:%.*]] = acc.private varPtr([[FP]] : memref<i64>) recipe(@privatization_memref_i64)
// CHECK: acc.loop private([[LP]] : memref<i64>)
// CHECK-NOT: acc.loop private([[LP]] : memref<i64>, [[LP]] : memref<i64>)

func.func @test_already_private_not_duplicated(%n : memref<i64>) {
  %c0 = arith.constant 0 : i64
  %c10 = arith.constant 10 : i64
  %c1 = arith.constant 1 : i64
  %fp = acc.firstprivate varPtr(%n : memref<i64>)
          recipe(@firstprivatization_memref_i64) -> memref<i64>
          {name = ""}
  acc.parallel firstprivate(%fp : memref<i64>) {
    // The loop already has an explicit private clause for %fp.
    %lp = acc.private varPtr(%fp : memref<i64>)
            recipe(@privatization_memref_i64) -> memref<i64>
            {name = ""}
    acc.loop private(%lp : memref<i64>)
             control(%iv : i64) = (%c0 : i64) to (%c10 : i64) step (%c1 : i64) {
      %val = memref.load %lp[] : memref<i64>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}
