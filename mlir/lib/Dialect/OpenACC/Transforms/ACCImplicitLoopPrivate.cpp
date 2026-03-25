//===- ACCImplicitLoopPrivate.cpp - Implicit privatization for acc loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds implicit firstprivate clauses to scalar variables used inside
// acc.loop constructs that are not already covered by an explicit private,
// firstprivate, or reduction clause.
//
// For each scalar variable that is live-in to an acc.loop region, an
// acc.firstprivate op is created before the loop and the loop's
// firstprivateOperands list is updated. The private copy is initialized from
// the value in the enclosing scope, giving firstprivate semantics.
//
// Requirements:
// - Variables must implement acc::MappableType or acc::PointerLikeType so
//   that the pass can determine their type category.
// - An acc::OpenACCSupport analysis must be available (or the default is
//   used).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

/// Returns true if `val` is already covered by a private, firstprivate, or
/// reduction operand on `loopOp`.
static bool isAlreadyPrivatized(Value val, acc::LoopOp loopOp) {
  auto isVar = [&](Value operand) {
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

/// Returns true if `val` is a candidate for implicit privatization in an
/// acc.loop: it must be a pointer-like or mappable type representing a scalar
/// variable, and not already valid for use in the region without a clause.
static bool isCandidateForLoopPrivate(Value val, Region &loopRegion,
                                      acc::OpenACCSupport &accSupport) {
  // Must be a type that can be privatized.
  if (!acc::isPointerLikeType(val.getType()) &&
      !acc::isMappableType(val.getType()))
    return false;

  // If already coming from a data clause, no need to add another.
  if (isa_and_nonnull<ACC_DATA_ENTRY_OPS>(val.getDefiningOp()))
    return false;

  // If the value is already valid (e.g. device data), skip it.
  if (accSupport.isValidValueUse(val, loopRegion))
    return false;

  // Only privatize scalars (not aggregates like arrays).
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
  /// Generates a firstprivate recipe for `var` in `module`, reusing an
  /// existing recipe of the same name if one already exists.
  acc::FirstprivateRecipeOp
  generateFirstprivateRecipe(ModuleOp module, Value var, Location loc,
                             OpBuilder &builder,
                             acc::OpenACCSupport &accSupport);

  /// Processes a single acc.loop operation, adding implicit firstprivate
  /// clauses for qualifying scalar variables.
  void processLoopOp(ModuleOp module, acc::LoopOp loopOp,
                     acc::OpenACCSupport &accSupport);
};

acc::FirstprivateRecipeOp
ACCImplicitLoopPrivate::generateFirstprivateRecipe(
    ModuleOp module, Value var, Location loc, OpBuilder &builder,
    acc::OpenACCSupport &accSupport) {
  Type type = var.getType();
  std::string recipeName =
      accSupport.getRecipeName(acc::RecipeKind::firstprivate_recipe, type, var);

  if (auto existing = module.lookupSymbol<acc::FirstprivateRecipeOp>(recipeName))
    return existing;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto recipe = acc::FirstprivateRecipeOp::createAndPopulate(
      builder, loc, recipeName, type);
  if (!recipe.has_value()) {
    accSupport.emitNYI(loc, "implicit loop firstprivate");
    return nullptr;
  }
  return recipe.value();
}

void ACCImplicitLoopPrivate::processLoopOp(ModuleOp module,
                                           acc::LoopOp loopOp,
                                           acc::OpenACCSupport &accSupport) {
  Region &loopRegion = loopOp.getRegion();

  // 1) Collect live-in values.
  SetVector<Value> liveInValues;
  getUsedValuesDefinedAbove(loopRegion, liveInValues);

  // 2) Filter to candidates that need implicit privatization.
  SmallVector<Value> candidates;
  for (Value val : liveInValues) {
    if (isAlreadyPrivatized(val, loopOp))
      continue;
    if (isCandidateForLoopPrivate(val, loopRegion, accSupport))
      candidates.push_back(val);
  }

  if (candidates.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "== ACCImplicitLoopPrivate: processing ==\n"
                           << loopOp << "\n");

  OpBuilder builder(loopOp);
  Location loc = loopOp.getLoc();
  SmallVector<Value> newFirstprivateOps;

  // 3) For each candidate, create an acc.firstprivate op before the loop.
  for (Value var : candidates) {
    std::string varName = accSupport.getVariableName(var);
    auto fpOp = acc::FirstprivateOp::create(builder, loc, var,
                                            /*structured=*/true,
                                            /*implicit=*/true, varName);
    LLVM_DEBUG(llvm::dbgs() << "  Created firstprivate for " << var << ": "
                             << fpOp << "\n");
    newFirstprivateOps.push_back(fpOp.getResult());
  }

  // 4) Replace uses of the original variables inside the loop with the
  //    result of the firstprivate ops.
  for (Value fpResult : newFirstprivateOps) {
    Value var = acc::getVar(fpResult.getDefiningOp());
    replaceAllUsesInRegionWith(var, fpResult, loopRegion);
  }

  // 5) Generate firstprivate recipes and attach them to the ops.
  for (Value fpResult : newFirstprivateOps) {
    auto fpOp = fpResult.getDefiningOp<acc::FirstprivateOp>();
    Value var = acc::getVar(fpOp);
    auto recipe =
        generateFirstprivateRecipe(module, var, loc, builder, accSupport);
    if (recipe)
      fpOp.setRecipeAttr(
          SymbolRefAttr::get(module->getContext(), recipe.getSymName()));
  }

  // 6) Add the new firstprivate operands to the loop op.
  for (Value fpResult : newFirstprivateOps)
    loopOp.getFirstprivateOperandsMutable().append(fpResult);
}

void ACCImplicitLoopPrivate::runOnOperation() {
  ModuleOp module = this->getOperation();
  acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();

  module.walk([&](acc::LoopOp loopOp) {
    processLoopOp(module, loopOp, accSupport);
  });
}

} // namespace
