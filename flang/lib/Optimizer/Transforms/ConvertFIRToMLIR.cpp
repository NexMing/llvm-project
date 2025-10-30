//===-- ConvertFIRToMLIR.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_CONVERTFIRTOMLIRPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class ConvertFIRToMLIRPass
    : public fir::impl::ConvertFIRToMLIRPassBase<ConvertFIRToMLIRPass> {
public:
  void runOnOperation() override;
};

/// TODO: how do we want to enforce this in MLIR? Can we manipulate the fast
/// math flags?
struct FIRNoReassocOpLowering
    : public mlir::OpConversionPattern<fir::NoReassocOp> {
  using mlir::OpConversionPattern<fir::NoReassocOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(fir::NoReassocOp noreassoc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(noreassoc, adaptor.getOperands()[0]);
    return mlir::success();
  }
};

class FIRLoadOpLowering : public mlir::OpConversionPattern<fir::LoadOp> {
public:
  using mlir::OpConversionPattern<fir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getMemref(),
                                                      mlir::ValueRange{});
    return mlir::success();
  }
};

class FIRStoreOpLowering : public mlir::OpConversionPattern<fir::StoreOp> {
public:
  using mlir::OpConversionPattern<fir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), mlir::ValueRange{});
    return mlir::success();
  }
};

class FIRConvertOpLowering : public mlir::OpConversionPattern<fir::ConvertOp> {
public:
  using mlir::OpConversionPattern<fir::ConvertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto val = adaptor.getValue();
    auto valType = val.getType();
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (valType.isIndex() || resultType.isIndex()) {
      rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, resultType,
                                                            val);
    } else if (valType.isInteger() && resultType.isInteger()) {
      if (valType.getIntOrFloatBitWidth() <
          resultType.getIntOrFloatBitWidth()) {
        rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(op, resultType, val);
      } else {
        rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, resultType, val);
      }
    } else {
      return mlir::failure();
    }

    return mlir::success();
  }
};

class FIRAllocOpLowering : public mlir::OpConversionPattern<fir::AllocaOp> {
public:
  using mlir::OpConversionPattern<fir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto allocaType = converter->convertType(op.getAllocatedType());
    auto resultType = converter->convertType(op.getType());
    auto allocaOp = mlir::memref::AllocaOp::create(
        rewriter, op.getLoc(), mlir::MemRefType::get({}, allocaType));
    allocaOp->setAttrs(op->getAttrs());
    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, resultType, allocaOp);
    return mlir::success();
  }
};

static mlir::Value makeFIRMemref(mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Location loc, mlir::Value memref,
                                 mlir::ValueRange shape) {
  auto metadata =
      mlir::memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  auto base = metadata.getBaseBuffer();
  auto offset = metadata.getOffset();
  size_t rank = shape.size();
  auto sizes = llvm::to_vector_of<mlir::OpFoldResult>(shape);
  mlir::SmallVector<mlir::OpFoldResult> strides;

  mlir::Value stride = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  for (size_t i = 0; i < rank; ++i) {
    strides.push_back(stride);
    stride = mlir::arith::MulIOp::create(rewriter, loc, stride, shape[i]);
  }

  return mlir::memref::ReinterpretCastOp::create(rewriter, loc, base, offset,
                                                 sizes, strides);
}

class FIRXEmboxLowering : public mlir::OpConversionPattern<fir::cg::XEmboxOp> {
public:
  using mlir::OpConversionPattern<fir::cg::XEmboxOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XEmboxOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<fir::ReferenceType>(op.getMemref().getType()) ||
        !op.getSlice().empty() || !op.getShift().empty())
      return mlir::failure();

    mlir::Value memref = makeFIRMemref(rewriter, op.getLoc(),
                                       adaptor.getMemref(), adaptor.getShape());

    rewriter.replaceOp(op, {memref});

    return mlir::success();
  }
};

class FIRXArrayCoorOpLowering
    : public mlir::OpConversionPattern<fir::cg::XArrayCoorOp> {
public:
  using mlir::OpConversionPattern<fir::cg::XArrayCoorOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XArrayCoorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<fir::ReferenceType>(op.getMemref().getType()))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    auto resultType = mlir::cast<mlir::MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    mlir::Value memref =
        makeFIRMemref(rewriter, loc, adaptor.getMemref(), adaptor.getShape());

    auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto offsets = llvm::map_to_vector(
        adaptor.getIndices(), [&](mlir::Value idx) -> mlir::OpFoldResult {
          if (idx.getType().isInteger())
            idx = mlir::arith::IndexCastOp::create(
                rewriter, loc, rewriter.getIndexType(), idx);

          assert(idx.getType().isIndex() && "expected index type");
          idx = mlir::arith::SubIOp::create(rewriter, loc, idx, one);
          return idx;
        });
    mlir::SmallVector<mlir::OpFoldResult> ones(op.getRank(),
                                               rewriter.getIndexAttr(1));

    auto subview = mlir::memref::SubViewOp::create(rewriter, loc, resultType,
                                                   memref, offsets, ones, ones);

    rewriter.replaceOp(op, mlir::ValueRange{subview});
    return mlir::success();
  }
};

class FuncOpLowering : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FunctionType fnType = op.getFunctionType();
    if (fnType.getNumResults() > 1)
      return mlir::failure();

    mlir::TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    for (auto [idx, argType] : enumerate(fnType.getInputs())) {
      mlir::Type convertedType = convertFuncType(argType);

      if (!convertedType)
        return mlir::failure();

      signatureConverter.addInputs(idx, convertedType);
    }

    mlir::SmallVector<mlir::Type> resultTypes;
    for (auto resType : fnType.getResults()) {
      mlir::Type convertedType = convertFuncType(resType);

      if (!convertedType)
        return mlir::failure();

      resultTypes.push_back(convertedType);
    }

    //  Create a legal func op.
    auto newFuncOp = mlir::func::FuncOp::create(
        rewriter, op.getLoc(), op.getName(),
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                 resultTypes));

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() != op.getFunctionTypeAttrName() &&
          namedAttr.getName() != mlir::SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::Type convertFuncType(mlir::Type type) const {
    mlir::Type convertedType = getTypeConverter()->convertType(type);
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(convertedType)) {
      convertedType = mlir::MemRefType::get({}, memrefType.getElementType());
    }
    return convertedType;
  }
};

} // namespace

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([](mlir::Type ty) { return ty; });
  converter.addConversion([&](fir::ReferenceType ty) {
    auto eleTy = ty.getElementType();
    if (auto sequenceTy = mlir::dyn_cast<fir::SequenceType>(eleTy))
      eleTy = sequenceTy.getElementType();

    auto layout = mlir::StridedLayoutAttr::get(ty.getContext(),
                                               mlir::ShapedType::kDynamic, {});
    return mlir::MemRefType::get({}, converter.convertType(eleTy), layout);
  });
  converter.addConversion([&](fir::BaseBoxType ty) {
    mlir::SmallVector<int64_t> shape;
    auto eleTy = ty.getEleTy();
    if (auto sequenceTy = mlir::dyn_cast<fir::SequenceType>(eleTy)) {
      llvm::append_range(shape, sequenceTy.getShape());
      eleTy = sequenceTy.getElementType();
    }

    auto layout = mlir::StridedLayoutAttr::get(ty.getContext(),
                                               mlir::ShapedType::kDynamic, {});
    return mlir::MemRefType::get(shape, converter.convertType(eleTy), layout);
  });

  // Use UnrealizedConversionCast as the bridge so that we don't need to pull
  // in patterns for other dialects.
  auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
    if (inputs.size() == 1 &&
        mlir::isa<mlir::BaseMemRefType>(inputs[0].getType()) &&
        mlir::isa<mlir::BaseMemRefType>(type)) {
      return mlir::memref::CastOp::create(builder, loc, type, inputs[0]);
    }

    auto cast =
        mlir::UnrealizedConversionCastOp::create(builder, loc, type, inputs);
    return cast.getResult(0);
  };

  converter.addSourceMaterialization(addUnrealizedCast);
  converter.addTargetMaterialization(addUnrealizedCast);
  return converter;
}

void ConvertFIRToMLIRPass::runOnOperation() {
  mlir::MLIRContext *ctx = &getContext();
  mlir::ModuleOp theModule = getOperation();
  mlir::TypeConverter converter = prepareTypeConverter();
  mlir::RewritePatternSet patterns(&getContext());
  mlir::ConversionTarget target(getContext());

  patterns.add<FIRNoReassocOpLowering, FIRAllocOpLowering, FIRLoadOpLowering,
               FIRStoreOpLowering, FIRConvertOpLowering, FIRXEmboxLowering,
               FIRXArrayCoorOpLowering, FuncOpLowering>(converter, ctx);

  target.addIllegalOp<fir::AllocaOp, fir::LoadOp, fir::StoreOp, fir::ConvertOp,
                      fir::cg::XArrayCoorOp, fir::cg::XEmboxOp,
                      fir::NoReassocOp>();
  target.addDynamicallyLegalOp<mlir::func::FuncOp>([](mlir::func::FuncOp op) {
    return !llvm::any_of(llvm::concat<const mlir::Type>(op.getResultTypes(),
                                                        op.getArgumentTypes()),
                         [&](const mlir::Type ty) {
                           return mlir::isa<fir::FIROpsDialect>(
                               ty.getDialect());
                         });
  });

  target.addLegalDialect<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect, mlir::affine::AffineDialect>();

  if (mlir::failed(mlir::applyPartialConversion(theModule, target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}
