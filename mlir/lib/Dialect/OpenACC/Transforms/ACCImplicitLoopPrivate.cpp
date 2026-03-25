//===- ACCImplicitLoopPrivate.cpp - Implicit private for acc.loop ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds implicit 'private' clauses to acc.loop constructs for scalar
// variables that are already privatized in an enclosing construct.
//
// Per the OpenACC specification, 'firstprivate' is only valid on compute
// constructs (acc.parallel, acc.serial, acc.kernels), not on acc.loop.
// This pass assumes that scalar variables have already been mapped with
// 'firstprivate' on an enclosing acc.parallel (or with 'private' on an
// enclosing acc.loop), and adds a corresponding 'private' clause to the
// acc.loop that uses the enclosing construct's privatized value as the
// source (varPtr).
//
// The pass processes acc.loop operations in pre-order (outer before inner),
// so nested loops correctly reference the outer loop's private result.
//
// Requirements:
// - Variables must implement acc::MappableType or acc::PointerLikeType.
// - An acc::OpenACCSupport analysis must be available (or default is used).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCIMPLICITLOOPPRIVATE
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-implicit-loop-private"

using namespace mlir;

namespace {

/// Returns true if `val` is already covered by a private or reduction operand
/// on `loopOp` (firstprivate is also checked since the dialect allows it as
/// an extension, but this pass only generates private).
static bool isAlreadyPrivatized(Value val, acc::LoopOp loopOp) {
  auto isVar = [&](Value operand) -> bool {
    if (Operation *defOp = operand.getDefiningOp())
      if (Value var = acc::getVar(defOp))
        return var == val;
    return false;
  };
  for (Value operand : loopOp.getPrivateOperands())
    if (isVar(operand))
      return true;
  for (Value operand : loopOp.getFirstprivateOperands())
    if (isVar(operand))
      return true;
  for (Value operand : loopOp.getReductionOperands())
    if (isVar(operand))
      return true;
  return false;
}

/// Returns true if `val` should receive an implicit private clause on the
/// surrounding acc.loop. The value must be:
///   1. A result of acc.firstprivate or acc.private from an enclosing
///      construct (parallel or outer loop) — this is the "upper level" value.
///   2. A scalar type (not an aggregate like an array).
///   3. Not already covered by a private/firstprivate/reduction on this loop.
static bool isCandidateForLoopPrivate(Value val, acc::LoopOp loopOp) {
  // Only values coming from acc.firstprivate (enclosing parallel) or
  // acc.private (enclosing outer loop) are candidates. These represent
  // scalars that have already been privatized at an upper level.
  if (!isa_and_nonnull<acc::FirstprivateOp, acc::PrivateOp>(
          val.getDefiningOp()))
    return false;

  // Must not already be handled by an explicit clause on this loop.
  if (isAlreadyPrivatized(val, loopOp))
    return false;

  // Only privatize scalars, not aggregates (arrays, structs, etc.).
  acc::VariableTypeCategory typeCategory = acc::getTypeCategory(val);
  return acc::bitEnumContainsAny(typeCategory,
                                 acc::VariableTypeCategory::scalar);
}

class ACCImplicitLoopPrivate
    : public acc::impl::ACCImplicitLoopPrivateBase<ACCImplicitLoopPrivate> {
public:
  using acc::impl::ACCImplicitLoopPrivateBase<
      ACCImplicitLoopPrivate>::ACCImplicitLoopPrivateBase;

  void runOnOperation() override;

private:
  /// Gets or creates an acc.firstprivate.recipe for the given variable type.
  /// The recipe is looked up by name in the module; if absent it is created.
  acc::FirstprivateRecipeOp
  getOrCreateFirstprivateRecipe(ModuleOp module, Value var, Location loc,
                                OpBuilder &builder,
                                acc::OpenACCSupport &accSupport);

  /// Gets or creates an acc.private.recipe derived from the given firstprivate
  /// recipe. The private recipe uses the firstprivate recipe's init region
  /// (which correctly handles e.g. dynamic-shape allocation), giving the
  /// loop-private copy the right shape initialized from the upper-level value.
  acc::PrivateRecipeOp
  getOrCreatePrivateRecipe(ModuleOp module, Value var, Location loc,
                           OpBuilder &builder,
                           acc::FirstprivateRecipeOp firstprivRecipe,
                           acc::OpenACCSupport &accSupport);

  /// Processes a single acc.loop, adding implicit private clauses for scalar
  /// variables that are live-in and come from an enclosing firstprivate or
  /// private op.
  void processLoopOp(ModuleOp module, acc::LoopOp loopOp,
                     acc::OpenACCSupport &accSupport);
};

acc::FirstprivateRecipeOp ACCImplicitLoopPrivate::getOrCreateFirstprivateRecipe(
    ModuleOp module, Value var, Location loc, OpBuilder &builder,
    acc::OpenACCSupport &accSupport) {
  Type type = var.getType();
  std::string recipeName =
      accSupport.getRecipeName(acc::RecipeKind::firstprivate_recipe, type, var);

  if (auto existing =
          module.lookupSymbol<acc::FirstprivateRecipeOp>(recipeName))
    return existing;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto recipe =
      acc::FirstprivateRecipeOp::createAndPopulate(builder, loc, recipeName, type);
  if (!recipe.has_value()) {
    accSupport.emitNYI(loc, "implicit loop private (firstprivate recipe)");
    return nullptr;
  }
  return recipe.value();
}

acc::PrivateRecipeOp ACCImplicitLoopPrivate::getOrCreatePrivateRecipe(
    ModuleOp module, Value var, Location loc, OpBuilder &builder,
    acc::FirstprivateRecipeOp firstprivRecipe,
    acc::OpenACCSupport &accSupport) {
  Type type = var.getType();
  // Use a distinct name to indicate this private recipe comes from a
  // firstprivate recipe (important for dynamic-shaped types).
  std::string recipeName =
      accSupport.getRecipeName(acc::RecipeKind::private_recipe, type, var);

  if (auto existing = module.lookupSymbol<acc::PrivateRecipeOp>(recipeName))
    return existing;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  // Create the private recipe using the firstprivate recipe's init region.
  // This ensures the private copy is allocated with the same strategy as
  // the firstprivate copy (e.g., matching dynamic dimensions).
  auto recipe = acc::PrivateRecipeOp::createAndPopulate(builder, loc,
                                                        recipeName,
                                                        firstprivRecipe);
  if (!recipe.has_value()) {
    accSupport.emitNYI(loc, "implicit loop private recipe");
    return nullptr;
  }
  return recipe.value();
}

void ACCImplicitLoopPrivate::processLoopOp(ModuleOp module,
                                           acc::LoopOp loopOp,
                                           acc::OpenACCSupport &accSupport) {
  Region &loopRegion = loopOp.getRegion();

  // 1) Collect all values defined outside the loop region that are used inside.
  SetVector<Value> liveInValues;
  getUsedValuesDefinedAbove(loopRegion, liveInValues);

  // 2) Filter to candidates: scalar values from acc.firstprivate or acc.private
  //    in enclosing constructs that are not yet in the loop's private operands.
  SmallVector<Value> candidates;
  for (Value val : liveInValues) {
    if (isCandidateForLoopPrivate(val, loopOp))
      candidates.push_back(val);
  }

  if (candidates.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "== ACCImplicitLoopPrivate: processing ==\n"
                           << loopOp << "\n");

  // Insert new acc.private ops just before the acc.loop (inside the enclosing
  // construct's region).
  OpBuilder builder(loopOp);
  Location loc = loopOp.getLoc();

  SmallVector<Value> newPrivateResults;

  // 3) For each candidate, create an acc.private op with varPtr pointing to
  //    the enclosing firstprivate/private result (the "upper level value").
  for (Value upperVal : candidates) {
    // Get the variable name from the upper-level op's varPtr.
    Value origVar = acc::getVar(upperVal.getDefiningOp());
    std::string varName = accSupport.getVariableName(origVar);

    // Get or create a firstprivate recipe for the type of the upper value
    // (which is the privatized type at this level).
    auto firstprivRecipe =
        getOrCreateFirstprivateRecipe(module, upperVal, loc, builder, accSupport);
    if (!firstprivRecipe)
      continue;

    // Get or create a private recipe derived from the firstprivate recipe.
    auto privRecipe = getOrCreatePrivateRecipe(module, upperVal, loc, builder,
                                               firstprivRecipe, accSupport);
    if (!privRecipe)
      continue;

    // Create acc.private varPtr(<upper_val>) — the upper-level firstprivate
    // or private result IS the source variable for this loop's private copy.
    auto privOp = acc::PrivateOp::create(builder, loc, upperVal,
                                         /*structured=*/true,
                                         /*implicit=*/true, varName);
    privOp.setRecipeAttr(
        SymbolRefAttr::get(module->getContext(), privRecipe.getSymName()));

    LLVM_DEBUG(llvm::dbgs()
               << "  Created loop private for " << upperVal << ": " << privOp
               << "\n");

    newPrivateResults.push_back(privOp.getResult());
  }

  if (newPrivateResults.empty())
    return;

  // 4) Replace uses of the upper-level values inside the loop with the new
  //    private results.
  for (Value privResult : newPrivateResults) {
    Value upperVal = acc::getVar(privResult.getDefiningOp());
    replaceAllUsesInRegionWith(upperVal, privResult, loopRegion);
  }

  // 5) Add the new private results to the loop's privateOperands.
  for (Value privResult : newPrivateResults)
    loopOp.getPrivateOperandsMutable().append(privResult);
}

void ACCImplicitLoopPrivate::runOnOperation() {
  ModuleOp module = this->getOperation();
  acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();

  // Walk in pre-order: outer loops are visited before inner loops.
  // This ensures that when we process an inner loop, the outer loop has
  // already had its private op added, so the inner loop sees the outer
  // private result as a live-in candidate.
  module.walk<WalkOrder::PreOrder>([&](acc::LoopOp loopOp) {
    processLoopOp(module, loopOp, accSupport);
  });
}

} // namespace
