// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.18

package kernels

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// CountMode controls the behavior of the count aggregate kernel.
type CountMode int8

const (
	// CountOnlyValid counts only non-null values. This is the default.
	CountOnlyValid CountMode = iota
	// CountOnlyNull counts only null values.
	CountOnlyNull
	// CountAll counts both non-null and null values.
	CountAll
)

// CountOptions controls count aggregate kernel behavior.
//
// By default, only non-null values are counted.
type CountOptions struct {
	Mode CountMode `compute:"mode"`
}

func (CountOptions) TypeName() string { return "CountOptions" }

// ScalarAggregateOptions controls the behavior of the scalar aggregate
// functions.
type ScalarAggregateOptions struct {
	// SkipNulls controls whether nulls are ignored (the default) or whether a
	// single null causes the output to be null.
	SkipNulls bool `compute:"skip_nulls"`
	// MinCount is the minimum number of non-null values required to produce a
	// non-null result. If fewer are present the result is null.
	MinCount uint32 `compute:"min_count"`
}

func (ScalarAggregateOptions) TypeName() string { return "ScalarAggregateOptions" }

// DefaultScalarAggregateOptions returns the default scalar aggregate options
// (skip nulls, min_count zero), matching the C++ defaults.
func DefaultScalarAggregateOptions() ScalarAggregateOptions {
	return ScalarAggregateOptions{SkipNulls: true}
}

// ScalarAggregator is the interface implemented by scalar aggregate kernel
// state objects. It mirrors the C++ ScalarAggregator.
type ScalarAggregator interface {
	Consume(*exec.KernelCtx, *exec.ExecSpan) error
	Merge(*exec.KernelCtx, exec.KernelState) error
	Finalize(*exec.KernelCtx) (scalar.Scalar, error)
}

func scalarAggConsume(ctx *exec.KernelCtx, span *exec.ExecSpan) error {
	agg, ok := ctx.State.(ScalarAggregator)
	if !ok {
		return fmt.Errorf("%w: invalid scalar aggregate kernel state %T", arrow.ErrInvalid, ctx.State)
	}
	return agg.Consume(ctx, span)
}

func scalarAggMerge(ctx *exec.KernelCtx, src exec.KernelState, dst *exec.KernelState) error {
	agg, ok := (*dst).(ScalarAggregator)
	if !ok {
		return fmt.Errorf("%w: invalid scalar aggregate kernel state %T", arrow.ErrInvalid, *dst)
	}
	return agg.Merge(ctx, src)
}

func scalarAggFinalize(ctx *exec.KernelCtx) (scalar.Scalar, error) {
	agg, ok := ctx.State.(ScalarAggregator)
	if !ok {
		return nil, fmt.Errorf("%w: invalid scalar aggregate kernel state %T", arrow.ErrInvalid, ctx.State)
	}
	return agg.Finalize(ctx)
}

func parseScalarAggOptions(opts any) (ScalarAggregateOptions, error) {
	switch value := opts.(type) {
	case nil:
		return DefaultScalarAggregateOptions(), nil
	case ScalarAggregateOptions:
		return value, nil
	case *ScalarAggregateOptions:
		if value == nil {
			return DefaultScalarAggregateOptions(), nil
		}
		return *value, nil
	default:
		return ScalarAggregateOptions{}, fmt.Errorf("%w: attempted to initialize scalar aggregate from invalid function options", arrow.ErrInvalid)
	}
}

func parseCountOptions(opts any) (CountOptions, error) {
	switch value := opts.(type) {
	case nil:
		return CountOptions{}, nil
	case CountOptions:
		return value, nil
	case *CountOptions:
		if value == nil {
			return CountOptions{}, nil
		}
		return *value, nil
	default:
		return CountOptions{}, fmt.Errorf("%w: attempted to initialize count from invalid function options", arrow.ErrInvalid)
	}
}

// valueNullCount returns the number of null values in the ExecValue, which is
// expected to have a length of `length` elements.
func valueNullCount(val *exec.ExecValue, length int64) int64 {
	if val.IsScalar() {
		if val.Scalar.IsValid() {
			return 0
		}
		return length
	}

	arr := &val.Array
	if arr.Type.ID() == arrow.NULL {
		// Null arrays have no validity bitmap but every value is null.
		return length
	}
	if arr.Nulls == 0 || arr.Buffers[0].Buf == nil {
		return 0
	}
	if arr.Nulls == arr.Len {
		return length
	}
	return arr.UpdateNullCount()
}

// fixedIter calls fn for each valid (non-null) value of a fixed-width array.
func fixedIter[T arrow.FixedWidthType](span *exec.ArraySpan, fn func(T)) {
	vals := exec.GetSpanValues[T](span, 1)
	if span.Nulls == 0 || span.Buffers[0].Buf == nil {
		for i := range vals {
			fn(vals[i])
		}
		return
	}
	if span.Nulls == span.Len {
		return
	}
	for i := range vals {
		if bitutil.BitIsSet(span.Buffers[0].Buf, int(span.Offset)+i) {
			fn(vals[i])
		}
	}
}

// boolIter calls fn for each valid (non-null) boolean value.
func boolIter(span *exec.ArraySpan, fn func(bool)) {
	data := span.Buffers[1].Buf
	if span.Nulls == 0 || span.Buffers[0].Buf == nil {
		for i := int64(0); i < span.Len; i++ {
			fn(bitutil.BitIsSet(data, int(span.Offset+i)))
		}
		return
	}
	if span.Nulls == span.Len {
		return
	}
	for i := int64(0); i < span.Len; i++ {
		if bitutil.BitIsSet(span.Buffers[0].Buf, int(span.Offset+i)) {
			fn(bitutil.BitIsSet(data, int(span.Offset+i)))
		}
	}
}

// validAt reports whether element i of the span is valid.
func validAt(span *exec.ArraySpan, i int64) bool {
	if span.Type.ID() == arrow.NULL {
		return false
	}
	if span.Nulls == 0 || span.Buffers[0].Buf == nil {
		return true
	}
	return bitutil.BitIsSet(span.Buffers[0].Buf, int(span.Offset+i))
}

// ----------------------------------------------------------------------
// Count implementation

type countState struct {
	mode  CountMode
	count int64
}

func initCount(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	opts, err := parseCountOptions(args.Options)
	if err != nil {
		return nil, err
	}
	return &countState{mode: opts.Mode}, nil
}

func (s *countState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if span.Len == 0 || len(span.Values) == 0 {
		return nil
	}
	nulls := valueNullCount(&span.Values[0], span.Len)
	switch s.mode {
	case CountAll:
		s.count += span.Len
	case CountOnlyNull:
		s.count += nulls
	default: // CountOnlyValid
		s.count += span.Len - nulls
	}
	return nil
}

func (s *countState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	other, ok := src.(*countState)
	if !ok {
		return fmt.Errorf("%w: invalid source kernel state for count", arrow.ErrInvalid)
	}
	s.count += other.count
	return nil
}

func (s *countState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	return scalar.NewInt64Scalar(s.count), nil
}

// ----------------------------------------------------------------------
// CountDistinct implementation

// distinctState counts distinct values by encoding each value as a string key.
// This works uniformly across fixed-width, boolean, binary and decimal types.
type distinctState struct {
	mode     CountMode
	seen     map[string]struct{}
	hasNulls bool
	forEach  func(*exec.ArraySpan, func(string))
}

func distinctKeyFunc(dt arrow.DataType) (func(*exec.ArraySpan, func(string)), error) {
	switch {
	case dt.ID() == arrow.BOOL:
		return func(span *exec.ArraySpan, fn func(string)) {
			boolIter(span, func(v bool) {
				if v {
					fn("\x01")
				} else {
					fn("\x00")
				}
			})
		}, nil
	case arrow.IsBinaryLike(dt.ID()):
		return func(span *exec.ArraySpan, fn func(string)) {
			offsets := exec.GetSpanOffsets[int32](span, 1)
			data := span.Buffers[2].Buf
			for i := int64(0); i < span.Len; i++ {
				if validAt(span, i) {
					fn(string(data[offsets[i]:offsets[i+1]]))
				}
			}
		}, nil
	case arrow.IsLargeBinaryLike(dt.ID()):
		return func(span *exec.ArraySpan, fn func(string)) {
			offsets := exec.GetSpanOffsets[int64](span, 1)
			data := span.Buffers[2].Buf
			for i := int64(0); i < span.Len; i++ {
				if validAt(span, i) {
					fn(string(data[offsets[i]:offsets[i+1]]))
				}
			}
		}, nil
	case arrow.IsFixedSizeBinary(dt.ID()):
		width := 0
		if fw, ok := dt.(arrow.FixedWidthDataType); ok {
			width = fw.BitWidth() / 8
		}
		if width == 0 {
			return nil, fmt.Errorf("%w: count_distinct not implemented for %s", arrow.ErrNotImplemented, dt)
		}
		return func(span *exec.ArraySpan, fn func(string)) {
			buf := span.Buffers[1].Buf
			for i := int64(0); i < span.Len; i++ {
				if validAt(span, i) {
					start := int(span.Offset+i) * width
					fn(string(buf[start : start+width]))
				}
			}
		}, nil
	case arrow.IsPrimitive(dt.ID()):
		fw, ok := dt.(arrow.FixedWidthDataType)
		if !ok {
			return nil, fmt.Errorf("%w: count_distinct not implemented for %s", arrow.ErrNotImplemented, dt)
		}
		width := fw.BitWidth() / 8
		return func(span *exec.ArraySpan, fn func(string)) {
			buf := span.Buffers[1].Buf
			for i := int64(0); i < span.Len; i++ {
				if validAt(span, i) {
					start := int(span.Offset+i) * width
					fn(string(buf[start : start+width]))
				}
			}
		}, nil
	default:
		return nil, fmt.Errorf("%w: count_distinct not implemented for %s", arrow.ErrNotImplemented, dt)
	}
}

func initCountDistinct(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	opts, err := parseCountOptions(args.Options)
	if err != nil {
		return nil, err
	}
	forEach, err := distinctKeyFunc(args.Inputs[0])
	if err != nil {
		return nil, err
	}
	return &distinctState{mode: opts.Mode, seen: make(map[string]struct{}), forEach: forEach}, nil
}

func (s *distinctState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	if valueNullCount(&span.Values[0], span.Len) > 0 {
		s.hasNulls = true
	}
	if span.Values[0].IsArray() {
		s.forEach(&span.Values[0].Array, func(k string) { s.seen[k] = struct{}{} })
	}
	return nil
}

func (s *distinctState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	other, ok := src.(*distinctState)
	if !ok {
		return fmt.Errorf("%w: invalid source kernel state for count_distinct", arrow.ErrInvalid)
	}
	for k := range other.seen {
		s.seen[k] = struct{}{}
	}
	s.hasNulls = s.hasNulls || other.hasNulls
	return nil
}

func (s *distinctState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	nonNulls := int64(len(s.seen))
	var nulls int64
	if s.hasNulls {
		nulls = 1
	}
	switch s.mode {
	case CountOnlyNull:
		return scalar.NewInt64Scalar(nulls), nil
	case CountAll:
		return scalar.NewInt64Scalar(nonNulls + nulls), nil
	default:
		return scalar.NewInt64Scalar(nonNulls), nil
	}
}

// ----------------------------------------------------------------------
// Kernel construction helpers

func aggKernel(in arrow.DataType, out exec.OutputType, init exec.KernelInitFn, ordered bool) exec.ScalarAggregateKernel {
	return exec.NewScalarAggregateKernel(
		[]exec.InputType{exec.NewExactInput(in)},
		out, init, scalarAggConsume, scalarAggMerge, scalarAggFinalize, ordered)
}

func aggKernelComputed(in arrow.DataType, resolver exec.TypeResolver, init exec.KernelInitFn, ordered bool) exec.ScalarAggregateKernel {
	return exec.NewScalarAggregateKernel(
		[]exec.InputType{exec.NewExactInput(in)},
		exec.NewComputedOutputType(resolver), init, scalarAggConsume, scalarAggMerge, scalarAggFinalize, ordered)
}

func identityOutType(_ *exec.KernelCtx, types []arrow.DataType) (arrow.DataType, error) {
	return types[0], nil
}

// appendDecimalKernels appends decimal128/256 kernels using a type-id matcher
// so any precision/scale matches.
func appendDecimalKernels(dst []exec.ScalarAggregateKernel, resolver exec.TypeResolver, init exec.KernelInitFn, ordered bool) []exec.ScalarAggregateKernel {
	dst = append(dst, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL128), resolver, init, ordered))
	dst = append(dst, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL256), resolver, init, ordered))
	return dst
}

// appendFixedSizeBinaryAggKernels appends a fixed-size-binary kernel using a
// type-id matcher so any byte-width matches.
func appendFixedSizeBinaryAggKernels(dst []exec.ScalarAggregateKernel, resolver exec.TypeResolver, init exec.KernelInitFn, ordered bool) []exec.ScalarAggregateKernel {
	dst = append(dst, aggKernelMatchedComputed(exec.SameTypeID(arrow.FIXED_SIZE_BINARY), resolver, init, ordered))
	return dst
}

func aggKernelMatched(matcher exec.TypeMatcher, out exec.OutputType, init exec.KernelInitFn, ordered bool) exec.ScalarAggregateKernel {
	return exec.NewScalarAggregateKernel(
		[]exec.InputType{exec.NewMatchedInput(matcher)},
		out, init, scalarAggConsume, scalarAggMerge, scalarAggFinalize, ordered)
}

func aggKernelMatchedComputed(matcher exec.TypeMatcher, resolver exec.TypeResolver, init exec.KernelInitFn, ordered bool) exec.ScalarAggregateKernel {
	return exec.NewScalarAggregateKernel(
		[]exec.InputType{exec.NewMatchedInput(matcher)},
		exec.NewComputedOutputType(resolver), init, scalarAggConsume, scalarAggMerge, scalarAggFinalize, ordered)
}

// ScalarAggregateKernels holds the kernels for all scalar aggregate functions.
type ScalarAggregateKernels struct {
	Count         []exec.ScalarAggregateKernel
	CountDistinct []exec.ScalarAggregateKernel
	Sum           []exec.ScalarAggregateKernel
	Mean          []exec.ScalarAggregateKernel
	Product       []exec.ScalarAggregateKernel
	MinMax        []exec.ScalarAggregateKernel
	Min           []exec.ScalarAggregateKernel
	Max           []exec.ScalarAggregateKernel
	Any           []exec.ScalarAggregateKernel
	All           []exec.ScalarAggregateKernel
	FirstLast     []exec.ScalarAggregateKernel
	First         []exec.ScalarAggregateKernel
	Last          []exec.ScalarAggregateKernel
	Index         []exec.ScalarAggregateKernel
	Variance      []exec.ScalarAggregateKernel
	StdDev        []exec.ScalarAggregateKernel
}

// GetScalarAggregateKernels returns the registered set of scalar aggregate
// kernels grouped by function.
func GetScalarAggregateKernels() ScalarAggregateKernels {
	var out ScalarAggregateKernels

	// count: any input -> int64
	out.Count = append(out.Count, exec.NewScalarAggregateKernel(
		[]exec.InputType{{Kind: exec.InputAny}},
		exec.NewOutputType(arrow.PrimitiveTypes.Int64),
		initCount, scalarAggConsume, scalarAggMerge, scalarAggFinalize, false))

	// count_distinct: supported input -> int64
	outCountDistinct := exec.NewOutputType(arrow.PrimitiveTypes.Int64)
	out.CountDistinct = append(out.CountDistinct,
		aggKernelMatched(exec.Primitive(), outCountDistinct, initCountDistinct, false),
		aggKernelMatched(exec.BinaryLike(), outCountDistinct, initCountDistinct, false),
		aggKernelMatched(exec.LargeBinaryLike(), outCountDistinct, initCountDistinct, false),
		aggKernelMatched(exec.FixedSizeBinaryLike(), outCountDistinct, initCountDistinct, false),
	)

	out.Sum = sumKernels()
	out.Mean = meanKernels()
	out.Product = productKernels()
	out.MinMax, out.Min, out.Max = minMaxKernels()
	out.Any = anyAllKernels(true)
	out.All = anyAllKernels(false)
	out.FirstLast, out.First, out.Last = firstLastKernels()
	out.Index = indexKernels()
	out.Variance, out.StdDev = varianceKernels()

	return out
}
