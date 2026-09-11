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
	"math"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/decimal256"
	"github.com/apache/arrow-go/v18/arrow/float16"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// widenDecimalType returns the maximum-precision decimal type with the same
// scale as the input, matching the C++ WidenDecimalToMaxPrecision helper.
func widenDecimalType(dt arrow.DataType) (arrow.DataType, error) {
	switch t := dt.(type) {
	case *arrow.Decimal128Type:
		return &arrow.Decimal128Type{Precision: 38, Scale: t.Scale}, nil
	case *arrow.Decimal256Type:
		return &arrow.Decimal256Type{Precision: 76, Scale: t.Scale}, nil
	default:
		return nil, fmt.Errorf("%w: expected a decimal type, got %s", arrow.ErrType, dt)
	}
}

func decimalWidenOutType(_ *exec.KernelCtx, types []arrow.DataType) (arrow.DataType, error) {
	return widenDecimalType(types[0])
}

// ----------------------------------------------------------------------
// Value accessors

// orderedAccessor provides type-erased access to the values of a supported
// array type for the ordering-based aggregations (min/max, first/last, index).
type orderedAccessor struct {
	childType  arrow.DataType
	iter       func(*exec.ArraySpan, func(any))
	at         func(*exec.ArraySpan, int64) any
	less       func(a, b any) bool
	equal      func(a, b any) bool
	toScalar   func(any) scalar.Scalar
	fromScalar func(scalar.Scalar) any
}

func numericAccessor[T arrow.FixedWidthType](dt arrow.DataType,
	less func(T, T) bool,
	toScalar func(T) scalar.Scalar,
	fromScalar func(scalar.Scalar) T) orderedAccessor {
	return orderedAccessor{
		childType:  dt,
		iter:       func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(v) }) },
		at:         func(s *exec.ArraySpan, i int64) any { return exec.GetSpanValues[T](s, 1)[i] },
		less:       func(a, b any) bool { return less(a.(T), b.(T)) },
		equal:      func(a, b any) bool { return a.(T) == b.(T) },
		toScalar:   func(v any) scalar.Scalar { return toScalar(v.(T)) },
		fromScalar: func(sc scalar.Scalar) any { return fromScalar(sc) },
	}
}

func orderedAccessorFor(dt arrow.DataType) (orderedAccessor, error) {
	switch dt.ID() {
	case arrow.NULL:
		return orderedAccessor{
			childType:  dt,
			iter:       func(*exec.ArraySpan, func(any)) {},
			at:         func(*exec.ArraySpan, int64) any { return nil },
			less:       func(a, b any) bool { return false },
			equal:      func(a, b any) bool { return false },
			toScalar:   func(any) scalar.Scalar { return scalar.MakeNullScalar(dt) },
			fromScalar: func(scalar.Scalar) any { return nil },
		}, nil
	case arrow.BOOL:
		return orderedAccessor{
			childType: dt,
			iter:      func(s *exec.ArraySpan, fn func(any)) { boolIter(s, func(v bool) { fn(v) }) },
			at: func(s *exec.ArraySpan, i int64) any {
				return bitAt(s, i)
			},
			less:       func(a, b any) bool { return !a.(bool) && b.(bool) },
			equal:      func(a, b any) bool { return a.(bool) == b.(bool) },
			toScalar:   func(v any) scalar.Scalar { return scalar.NewBooleanScalar(v.(bool)) },
			fromScalar: func(sc scalar.Scalar) any { return sc.(*scalar.Boolean).Value },
		}, nil
	case arrow.INT8:
		return numericAccessor[int8](dt, func(a, b int8) bool { return a < b }, func(v int8) scalar.Scalar { return scalar.NewInt8Scalar(v) }, func(sc scalar.Scalar) int8 { return sc.(*scalar.Int8).Value }), nil
	case arrow.INT16:
		return numericAccessor[int16](dt, func(a, b int16) bool { return a < b }, func(v int16) scalar.Scalar { return scalar.NewInt16Scalar(v) }, func(sc scalar.Scalar) int16 { return sc.(*scalar.Int16).Value }), nil
	case arrow.INT32:
		return numericAccessor[int32](dt, func(a, b int32) bool { return a < b }, func(v int32) scalar.Scalar { return scalar.NewInt32Scalar(v) }, func(sc scalar.Scalar) int32 { return sc.(*scalar.Int32).Value }), nil
	case arrow.INT64:
		return numericAccessor[int64](dt, func(a, b int64) bool { return a < b }, func(v int64) scalar.Scalar { return scalar.NewInt64Scalar(v) }, func(sc scalar.Scalar) int64 { return sc.(*scalar.Int64).Value }), nil
	case arrow.UINT8:
		return numericAccessor[uint8](dt, func(a, b uint8) bool { return a < b }, func(v uint8) scalar.Scalar { return scalar.NewUint8Scalar(v) }, func(sc scalar.Scalar) uint8 { return sc.(*scalar.Uint8).Value }), nil
	case arrow.UINT16:
		return numericAccessor[uint16](dt, func(a, b uint16) bool { return a < b }, func(v uint16) scalar.Scalar { return scalar.NewUint16Scalar(v) }, func(sc scalar.Scalar) uint16 { return sc.(*scalar.Uint16).Value }), nil
	case arrow.UINT32:
		return numericAccessor[uint32](dt, func(a, b uint32) bool { return a < b }, func(v uint32) scalar.Scalar { return scalar.NewUint32Scalar(v) }, func(sc scalar.Scalar) uint32 { return sc.(*scalar.Uint32).Value }), nil
	case arrow.UINT64:
		return numericAccessor[uint64](dt, func(a, b uint64) bool { return a < b }, func(v uint64) scalar.Scalar { return scalar.NewUint64Scalar(v) }, func(sc scalar.Scalar) uint64 { return sc.(*scalar.Uint64).Value }), nil
	case arrow.FLOAT16:
		return numericAccessor[float16.Num](dt,
			func(a, b float16.Num) bool { return a.Float32() < b.Float32() },
			func(v float16.Num) scalar.Scalar { return scalar.NewFloat16Scalar(v) },
			func(sc scalar.Scalar) float16.Num { return sc.(*scalar.Float16).Value }), nil
	case arrow.FLOAT32:
		return numericAccessor[float32](dt, func(a, b float32) bool { return a < b }, func(v float32) scalar.Scalar { return scalar.NewFloat32Scalar(v) }, func(sc scalar.Scalar) float32 { return sc.(*scalar.Float32).Value }), nil
	case arrow.FLOAT64:
		return numericAccessor[float64](dt, func(a, b float64) bool { return a < b }, func(v float64) scalar.Scalar { return scalar.NewFloat64Scalar(v) }, func(sc scalar.Scalar) float64 { return sc.(*scalar.Float64).Value }), nil
	case arrow.DATE32:
		return numericAccessor[arrow.Date32](dt, func(a, b arrow.Date32) bool { return a < b }, func(v arrow.Date32) scalar.Scalar { return scalar.NewDate32Scalar(v) }, func(sc scalar.Scalar) arrow.Date32 { return sc.(*scalar.Date32).Value }), nil
	case arrow.DATE64:
		return numericAccessor[arrow.Date64](dt, func(a, b arrow.Date64) bool { return a < b }, func(v arrow.Date64) scalar.Scalar { return scalar.NewDate64Scalar(v) }, func(sc scalar.Scalar) arrow.Date64 { return sc.(*scalar.Date64).Value }), nil
	case arrow.TIME32:
		return numericAccessor[arrow.Time32](dt, func(a, b arrow.Time32) bool { return a < b }, func(v arrow.Time32) scalar.Scalar { return scalar.NewTime32Scalar(v, dt) }, func(sc scalar.Scalar) arrow.Time32 { return sc.(*scalar.Time32).Value }), nil
	case arrow.TIME64:
		return numericAccessor[arrow.Time64](dt, func(a, b arrow.Time64) bool { return a < b }, func(v arrow.Time64) scalar.Scalar { return scalar.NewTime64Scalar(v, dt) }, func(sc scalar.Scalar) arrow.Time64 { return sc.(*scalar.Time64).Value }), nil
	case arrow.TIMESTAMP:
		return numericAccessor[arrow.Timestamp](dt, func(a, b arrow.Timestamp) bool { return a < b }, func(v arrow.Timestamp) scalar.Scalar { return scalar.NewTimestampScalar(v, dt) }, func(sc scalar.Scalar) arrow.Timestamp { return sc.(*scalar.Timestamp).Value }), nil
	case arrow.DURATION:
		return numericAccessor[arrow.Duration](dt, func(a, b arrow.Duration) bool { return a < b }, func(v arrow.Duration) scalar.Scalar { return scalar.NewDurationScalar(v, dt) }, func(sc scalar.Scalar) arrow.Duration { return sc.(*scalar.Duration).Value }), nil
	case arrow.INTERVAL_MONTHS:
		return numericAccessor[arrow.MonthInterval](dt, func(a, b arrow.MonthInterval) bool { return a < b }, func(v arrow.MonthInterval) scalar.Scalar { return scalar.NewMonthIntervalScalar(v) }, func(sc scalar.Scalar) arrow.MonthInterval { return sc.(*scalar.MonthInterval).Value }), nil
	case arrow.STRING, arrow.LARGE_STRING, arrow.BINARY, arrow.LARGE_BINARY:
		return binaryAccessor(dt)
	case arrow.FIXED_SIZE_BINARY:
		width := dt.(*arrow.FixedSizeBinaryType).ByteWidth
		return orderedAccessor{
			childType: dt,
			iter: func(s *exec.ArraySpan, fn func(any)) {
				buf := s.Buffers[1].Buf
				for i := int64(0); i < s.Len; i++ {
					if validAt(s, i) {
						start := int(s.Offset+i) * width
						fn(string(buf[start : start+width]))
					}
				}
			},
			at: func(s *exec.ArraySpan, i int64) any {
				start := int(s.Offset+i) * width
				return string(s.Buffers[1].Buf[start : start+width])
			},
			less:  func(a, b any) bool { return a.(string) < b.(string) },
			equal: func(a, b any) bool { return a.(string) == b.(string) },
			toScalar: func(v any) scalar.Scalar {
				return scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte(v.(string))), dt)
			},
			fromScalar: func(sc scalar.Scalar) any {
				return string(sc.(*scalar.FixedSizeBinary).Data())
			},
		}, nil
	case arrow.DECIMAL128:
		return numericAccessor[decimal128.Num](dt,
			func(a, b decimal128.Num) bool { return a.Cmp(b) < 0 },
			func(v decimal128.Num) scalar.Scalar { return scalar.NewDecimal128Scalar(v, dt) },
			func(sc scalar.Scalar) decimal128.Num { return sc.(*scalar.Decimal128).Value }), nil
	case arrow.DECIMAL256:
		return numericAccessor[decimal256.Num](dt,
			func(a, b decimal256.Num) bool { return a.Cmp(b) < 0 },
			func(v decimal256.Num) scalar.Scalar { return scalar.NewDecimal256Scalar(v, dt) },
			func(sc scalar.Scalar) decimal256.Num { return sc.(*scalar.Decimal256).Value }), nil
	default:
		return orderedAccessor{}, fmt.Errorf("%w: scalar aggregate not implemented for %s", arrow.ErrNotImplemented, dt)
	}
}

func binaryAccessor(dt arrow.DataType) (orderedAccessor, error) {
	large := arrow.IsLargeBinaryLike(dt.ID())
	sliceAt := func(s *exec.ArraySpan, i int64) string {
		if large {
			offsets := exec.GetSpanOffsets[int64](s, 1)
			return string(s.Buffers[2].Buf[offsets[i]:offsets[i+1]])
		}
		offsets := exec.GetSpanOffsets[int32](s, 1)
		return string(s.Buffers[2].Buf[offsets[i]:offsets[i+1]])
	}
	iter := func(s *exec.ArraySpan, fn func(any)) {
		if large {
			offsets := exec.GetSpanOffsets[int64](s, 1)
			data := s.Buffers[2].Buf
			for i := int64(0); i < s.Len; i++ {
				if validAt(s, i) {
					fn(string(data[offsets[i]:offsets[i+1]]))
				}
			}
			return
		}
		offsets := exec.GetSpanOffsets[int32](s, 1)
		data := s.Buffers[2].Buf
		for i := int64(0); i < s.Len; i++ {
			if validAt(s, i) {
				fn(string(data[offsets[i]:offsets[i+1]]))
			}
		}
	}
	return orderedAccessor{
		childType: dt,
		iter:      iter,
		at:        func(s *exec.ArraySpan, i int64) any { return sliceAt(s, i) },
		less:      func(a, b any) bool { return a.(string) < b.(string) },
		equal:     func(a, b any) bool { return a.(string) == b.(string) },
		toScalar: func(v any) scalar.Scalar {
			buf := memory.NewBufferBytes([]byte(v.(string)))
			return scalar.NewBinaryScalar(buf, dt)
		},
		fromScalar: func(sc scalar.Scalar) any {
			return string(sc.(scalar.BinaryScalar).Data())
		},
	}, nil
}

func bitAt(s *exec.ArraySpan, i int64) bool {
	return (s.Buffers[1].Buf[int(s.Offset+i)/8] & (1 << (uint(s.Offset+i) % 8))) != 0
}

// orderedTypes returns the types supported by min/max, first/last and index.
func orderedTypes() []arrow.DataType {
	var types []arrow.DataType
	types = append(types, arrow.Null, arrow.FixedWidthTypes.Boolean)
	types = append(types, signedIntTypes...)
	types = append(types, unsignedIntTypes...)
	types = append(types, arrow.FixedWidthTypes.Float16, arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64)
	types = append(types, arrow.FixedWidthTypes.Date32, arrow.FixedWidthTypes.Date64)
	types = append(types, arrow.FixedWidthTypes.Time32s, arrow.FixedWidthTypes.Time32ms,
		arrow.FixedWidthTypes.Time64us, arrow.FixedWidthTypes.Time64ns)
	types = append(types, arrow.FixedWidthTypes.Timestamp_s, arrow.FixedWidthTypes.Timestamp_ms,
		arrow.FixedWidthTypes.Timestamp_us, arrow.FixedWidthTypes.Timestamp_ns)
	types = append(types, arrow.FixedWidthTypes.Duration_s, arrow.FixedWidthTypes.Duration_ms,
		arrow.FixedWidthTypes.Duration_us, arrow.FixedWidthTypes.Duration_ns)
	types = append(types, arrow.FixedWidthTypes.MonthInterval)
	types = append(types, baseBinaryTypes...)
	return types
}

// ----------------------------------------------------------------------
// Sum

type sumState struct {
	opts     ScalarAggregateOptions
	outType  arrow.DataType
	acc      any
	add      func(any, any) any
	iter     func(*exec.ArraySpan, func(any))
	toScalar func(any) scalar.Scalar
	count    int64
	nulls    bool
}

func (s *sumState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.nulls = s.nulls || nulls > 0
	if !s.opts.SkipNulls && s.nulls {
		return nil
	}
	if v.IsArray() {
		s.iter(&v.Array, func(x any) { s.acc = s.add(s.acc, x) })
	}
	return nil
}

func (s *sumState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*sumState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for sum", arrow.ErrInvalid)
	}
	s.count += o.count
	s.acc = s.add(s.acc, o.acc)
	s.nulls = s.nulls || o.nulls
	return nil
}

func (s *sumState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if (!s.opts.SkipNulls && s.nulls) || s.count < int64(s.opts.MinCount) {
		return scalar.MakeNullScalar(s.outType), nil
	}
	return s.toScalar(s.acc), nil
}

func initSum(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	opts, err := parseScalarAggOptions(args.Options)
	if err != nil {
		return nil, err
	}
	return newSumState(args.Inputs[0], opts)
}

func newSumState(dt arrow.DataType, opts ScalarAggregateOptions) (*sumState, error) {
	switch dt.ID() {
	case arrow.NULL:
		return &sumState{opts: opts, outType: arrow.PrimitiveTypes.Int64,
			acc: int64(0), add: func(a, b any) any { return a.(int64) + b.(int64) },
			iter:     func(*exec.ArraySpan, func(any)) {},
			toScalar: func(a any) scalar.Scalar { return scalar.NewInt64Scalar(a.(int64)) }}, nil
	case arrow.BOOL:
		return &sumState{opts: opts, outType: arrow.PrimitiveTypes.Uint64,
			acc: uint64(0), add: func(a, b any) any { return a.(uint64) + b.(uint64) },
			iter: func(s *exec.ArraySpan, fn func(any)) {
				boolIter(s, func(v bool) {
					if v {
						fn(uint64(1))
					}
				})
			},
			toScalar: func(a any) scalar.Scalar { return scalar.NewUint64Scalar(a.(uint64)) }}, nil
	case arrow.INT8:
		return signedSum[int8](arrow.PrimitiveTypes.Int64, opts), nil
	case arrow.INT16:
		return signedSum[int16](arrow.PrimitiveTypes.Int64, opts), nil
	case arrow.INT32:
		return signedSum[int32](arrow.PrimitiveTypes.Int64, opts), nil
	case arrow.INT64:
		return signedSum[int64](arrow.PrimitiveTypes.Int64, opts), nil
	case arrow.UINT8:
		return unsignedSum[uint8](arrow.PrimitiveTypes.Uint64, opts), nil
	case arrow.UINT16:
		return unsignedSum[uint16](arrow.PrimitiveTypes.Uint64, opts), nil
	case arrow.UINT32:
		return unsignedSum[uint32](arrow.PrimitiveTypes.Uint64, opts), nil
	case arrow.UINT64:
		return unsignedSum[uint64](arrow.PrimitiveTypes.Uint64, opts), nil
	case arrow.FLOAT16:
		return floatSum[float16.Num](arrow.PrimitiveTypes.Float64, opts, func(v float16.Num) float64 { return float64(v.Float32()) }), nil
	case arrow.FLOAT32:
		return floatSum[float32](arrow.PrimitiveTypes.Float64, opts, func(v float32) float64 { return float64(v) }), nil
	case arrow.FLOAT64:
		return floatSum[float64](arrow.PrimitiveTypes.Float64, opts, func(v float64) float64 { return v }), nil
	case arrow.DECIMAL128:
		outType, err := widenDecimalType(dt)
		if err != nil {
			return nil, err
		}
		return &sumState{opts: opts, outType: outType,
			acc: decimal128.Num{},
			add: func(a, b any) any { return a.(decimal128.Num).Add(b.(decimal128.Num)) },
			iter: func(s *exec.ArraySpan, fn func(any)) {
				fixedIter[decimal128.Num](s, func(v decimal128.Num) { fn(v) })
			},
			toScalar: func(a any) scalar.Scalar {
				return scalar.NewDecimal128Scalar(a.(decimal128.Num), outType)
			}}, nil
	case arrow.DECIMAL256:
		outType, err := widenDecimalType(dt)
		if err != nil {
			return nil, err
		}
		return &sumState{opts: opts, outType: outType,
			acc: decimal256.Num{},
			add: func(a, b any) any { return a.(decimal256.Num).Add(b.(decimal256.Num)) },
			iter: func(s *exec.ArraySpan, fn func(any)) {
				fixedIter[decimal256.Num](s, func(v decimal256.Num) { fn(v) })
			},
			toScalar: func(a any) scalar.Scalar {
				return scalar.NewDecimal256Scalar(a.(decimal256.Num), outType)
			}}, nil
	default:
		return nil, fmt.Errorf("%w: sum not implemented for %s", arrow.ErrNotImplemented, dt)
	}
}

func signedSum[T arrow.IntType](out arrow.DataType, opts ScalarAggregateOptions) *sumState {
	return &sumState{opts: opts, outType: out,
		acc: int64(0), add: func(a, b any) any { return a.(int64) + b.(int64) },
		iter:     func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(int64(v)) }) },
		toScalar: func(a any) scalar.Scalar { return scalar.NewInt64Scalar(a.(int64)) }}
}

func unsignedSum[T arrow.UintType](out arrow.DataType, opts ScalarAggregateOptions) *sumState {
	return &sumState{opts: opts, outType: out,
		acc: uint64(0), add: func(a, b any) any { return a.(uint64) + b.(uint64) },
		iter:     func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(uint64(v)) }) },
		toScalar: func(a any) scalar.Scalar { return scalar.NewUint64Scalar(a.(uint64)) }}
}

func floatSum[T arrow.FixedWidthType](out arrow.DataType, opts ScalarAggregateOptions, conv func(T) float64) *sumState {
	return &sumState{opts: opts, outType: out,
		acc: float64(0), add: func(a, b any) any { return a.(float64) + b.(float64) },
		iter:     func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(conv(v)) }) },
		toScalar: func(a any) scalar.Scalar { return scalar.NewFloat64Scalar(a.(float64)) }}
}

func sumKernels() []exec.ScalarAggregateKernel {
	var out []exec.ScalarAggregateKernel
	out = append(out, aggKernel(arrow.Null, exec.NewOutputType(arrow.PrimitiveTypes.Int64), initSum, false))
	out = append(out, aggKernel(arrow.FixedWidthTypes.Boolean, exec.NewOutputType(arrow.PrimitiveTypes.Uint64), initSum, false))
	for _, dt := range signedIntTypes {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Int64), initSum, false))
	}
	for _, dt := range unsignedIntTypes {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Uint64), initSum, false))
	}
	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Float16, arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64} {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Float64), initSum, false))
	}
	out = append(out, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL128), decimalWidenOutType, initSum, false))
	out = append(out, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL256), decimalWidenOutType, initSum, false))
	return out
}

// ----------------------------------------------------------------------
// Mean

type meanState struct {
	opts     ScalarAggregateOptions
	outType  arrow.DataType
	sum      float64
	iter     func(*exec.ArraySpan, func(any))
	count    int64
	nulls    bool
	isNull   bool
	emptyVal float64
}

func (s *meanState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.nulls = s.nulls || nulls > 0
	if !s.opts.SkipNulls && s.nulls {
		return nil
	}
	if v.IsArray() {
		s.iter(&v.Array, func(x any) { s.sum += x.(float64) })
	}
	return nil
}

func (s *meanState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*meanState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for mean", arrow.ErrInvalid)
	}
	s.count += o.count
	s.sum += o.sum
	s.nulls = s.nulls || o.nulls
	return nil
}

func (s *meanState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if s.isNull {
		if s.opts.SkipNulls && s.opts.MinCount == 0 {
			return scalar.NewFloat64Scalar(s.emptyVal), nil
		}
		return scalar.MakeNullScalar(s.outType), nil
	}
	if (!s.opts.SkipNulls && s.nulls) || s.count < int64(s.opts.MinCount) {
		return scalar.MakeNullScalar(s.outType), nil
	}
	return scalar.NewFloat64Scalar(s.sum / float64(s.count)), nil
}

func initMean(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	opts, err := parseScalarAggOptions(args.Options)
	if err != nil {
		return nil, err
	}
	return newMeanState(args.Inputs[0], opts)
}

func newMeanState(dt arrow.DataType, opts ScalarAggregateOptions) (ScalarAggregator, error) {
	if dt.ID() == arrow.NULL {
		return &meanState{opts: opts, outType: arrow.PrimitiveTypes.Float64, isNull: true,
			iter: func(*exec.ArraySpan, func(any)) {}}, nil
	}
	switch dt.ID() {
	case arrow.DECIMAL128, arrow.DECIMAL256:
		return newDecimalMeanState(dt, opts)
	}
	base := &meanState{opts: opts, outType: arrow.PrimitiveTypes.Float64}
	switch dt.ID() {
	case arrow.BOOL:
		base.iter = func(s *exec.ArraySpan, fn func(any)) {
			boolIter(s, func(v bool) {
				if v {
					fn(float64(1))
				} else {
					fn(float64(0))
				}
			})
		}
	case arrow.INT8:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[int8](s, func(v int8) { fn(float64(v)) }) }
	case arrow.INT16:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[int16](s, func(v int16) { fn(float64(v)) }) }
	case arrow.INT32:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[int32](s, func(v int32) { fn(float64(v)) }) }
	case arrow.INT64:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[int64](s, func(v int64) { fn(float64(v)) }) }
	case arrow.UINT8:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[uint8](s, func(v uint8) { fn(float64(v)) }) }
	case arrow.UINT16:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[uint16](s, func(v uint16) { fn(float64(v)) }) }
	case arrow.UINT32:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[uint32](s, func(v uint32) { fn(float64(v)) }) }
	case arrow.UINT64:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[uint64](s, func(v uint64) { fn(float64(v)) }) }
	case arrow.FLOAT16:
		base.iter = func(s *exec.ArraySpan, fn func(any)) {
			fixedIter[float16.Num](s, func(v float16.Num) { fn(float64(v.Float32())) })
		}
	case arrow.FLOAT32:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[float32](s, func(v float32) { fn(float64(v)) }) }
	case arrow.FLOAT64:
		base.iter = func(s *exec.ArraySpan, fn func(any)) { fixedIter[float64](s, func(v float64) { fn(v) }) }
	default:
		return nil, fmt.Errorf("%w: mean not implemented for %s", arrow.ErrNotImplemented, dt)
	}
	return base, nil
}

func meanKernels() []exec.ScalarAggregateKernel {
	var out []exec.ScalarAggregateKernel
	out = append(out, aggKernel(arrow.Null, exec.NewOutputType(arrow.PrimitiveTypes.Float64), initMean, false))
	out = append(out, aggKernel(arrow.FixedWidthTypes.Boolean, exec.NewOutputType(arrow.PrimitiveTypes.Float64), initMean, false))
	for _, dt := range numericTypes {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Float64), initMean, false))
	}
	out = append(out, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL128), decimalWidenOutType, initMean, false))
	out = append(out, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL256), decimalWidenOutType, initMean, false))
	return out
}

// meanDecimalState accumulates decimal values and divides by the count with
// round-half-away-from-zero, matching the C++ decimal mean.
type meanDecimalState struct {
	opts    ScalarAggregateOptions
	outType arrow.DataType
	count   int64
	nulls   bool
	sum     any
	add     func(any, any) any
	iter    func(*exec.ArraySpan, func(any))
	divide  func(sum any, count int64) scalar.Scalar
}

func (s *meanDecimalState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.nulls = s.nulls || nulls > 0
	if !s.opts.SkipNulls && s.nulls {
		return nil
	}
	if v.IsArray() {
		s.iter(&v.Array, func(x any) { s.sum = s.add(s.sum, x) })
	}
	return nil
}

func (s *meanDecimalState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*meanDecimalState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for mean", arrow.ErrInvalid)
	}
	s.count += o.count
	s.sum = s.add(s.sum, o.sum)
	s.nulls = s.nulls || o.nulls
	return nil
}

func (s *meanDecimalState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if (!s.opts.SkipNulls && s.nulls) || s.count < int64(s.opts.MinCount) || s.count == 0 {
		return scalar.MakeNullScalar(s.outType), nil
	}
	return s.divide(s.sum, s.count), nil
}

func newDecimalMeanState(dt arrow.DataType, opts ScalarAggregateOptions) (ScalarAggregator, error) {
	outType, err := widenDecimalType(dt)
	if err != nil {
		return nil, err
	}
	switch dt.ID() {
	case arrow.DECIMAL128:
		return &meanDecimalState{opts: opts, outType: outType,
			sum: decimal128.Num{},
			add: func(a, b any) any { return a.(decimal128.Num).Add(b.(decimal128.Num)) },
			iter: func(s *exec.ArraySpan, fn func(any)) {
				fixedIter[decimal128.Num](s, func(v decimal128.Num) { fn(v) })
			},
			divide: func(sum any, count int64) scalar.Scalar {
				v := sum.(decimal128.Num)
				q, r := v.Div(decimal128.FromI64(count))
				r = r.Abs()
				if r.Add(r).Cmp(decimal128.FromI64(count)) >= 0 {
					if v.Sign() >= 0 {
						q = q.Add(decimal128.FromI64(1))
					} else {
						q = q.Sub(decimal128.FromI64(1))
					}
				}
				return scalar.NewDecimal128Scalar(q, outType)
			}}, nil
	default:
		return &meanDecimalState{opts: opts, outType: outType,
			sum: decimal256.Num{},
			add: func(a, b any) any { return a.(decimal256.Num).Add(b.(decimal256.Num)) },
			iter: func(s *exec.ArraySpan, fn func(any)) {
				fixedIter[decimal256.Num](s, func(v decimal256.Num) { fn(v) })
			},
			divide: func(sum any, count int64) scalar.Scalar {
				v := sum.(decimal256.Num)
				q, r := v.Div(decimal256.FromI64(count))
				r = r.Abs()
				if r.Add(r).Cmp(decimal256.FromI64(count)) >= 0 {
					if v.Sign() >= 0 {
						q = q.Add(decimal256.FromI64(1))
					} else {
						q = q.Sub(decimal256.FromI64(1))
					}
				}
				return scalar.NewDecimal256Scalar(q, outType)
			}}, nil
	}
}

// ----------------------------------------------------------------------
// Product

type productState struct {
	opts     ScalarAggregateOptions
	outType  arrow.DataType
	acc      any
	mul      func(any, any) any
	iter     func(*exec.ArraySpan, func(any))
	toScalar func(any) scalar.Scalar
	count    int64
	nulls    bool
	isNull   bool
	emptyAcc any
}

func (s *productState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.nulls = s.nulls || nulls > 0
	if !s.opts.SkipNulls && s.nulls {
		return nil
	}
	if v.IsArray() {
		s.iter(&v.Array, func(x any) { s.acc = s.mul(s.acc, x) })
	}
	return nil
}

func (s *productState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*productState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for product", arrow.ErrInvalid)
	}
	s.count += o.count
	s.acc = s.mul(s.acc, o.acc)
	s.nulls = s.nulls || o.nulls
	return nil
}

func (s *productState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if s.isNull {
		if s.opts.SkipNulls && s.opts.MinCount == 0 {
			return s.toScalar(s.emptyAcc), nil
		}
		return scalar.MakeNullScalar(s.outType), nil
	}
	if (!s.opts.SkipNulls && s.nulls) || s.count < int64(s.opts.MinCount) {
		return scalar.MakeNullScalar(s.outType), nil
	}
	return s.toScalar(s.acc), nil
}

func initProduct(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	opts, err := parseScalarAggOptions(args.Options)
	if err != nil {
		return nil, err
	}
	return newProductState(args.Inputs[0], opts)
}

func newProductState(dt arrow.DataType, opts ScalarAggregateOptions) (*productState, error) {
	switch dt.ID() {
	case arrow.NULL:
		return &productState{opts: opts, outType: arrow.PrimitiveTypes.Int64, isNull: true,
			acc: int64(1), emptyAcc: int64(1), mul: func(a, b any) any { return a.(int64) * b.(int64) },
			iter:     func(*exec.ArraySpan, func(any)) {},
			toScalar: func(a any) scalar.Scalar { return scalar.NewInt64Scalar(a.(int64)) }}, nil
	case arrow.BOOL:
		return &productState{opts: opts, outType: arrow.PrimitiveTypes.Uint64,
			acc: uint64(1), mul: func(a, b any) any { return a.(uint64) * b.(uint64) },
			iter: func(s *exec.ArraySpan, fn func(any)) {
				boolIter(s, func(v bool) {
					if v {
						fn(uint64(1))
					} else {
						fn(uint64(0))
					}
				})
			},
			toScalar: func(a any) scalar.Scalar { return scalar.NewUint64Scalar(a.(uint64)) }}, nil
	case arrow.INT8:
		return signedProduct[int8](opts), nil
	case arrow.INT16:
		return signedProduct[int16](opts), nil
	case arrow.INT32:
		return signedProduct[int32](opts), nil
	case arrow.INT64:
		return signedProduct[int64](opts), nil
	case arrow.UINT8:
		return unsignedProduct[uint8](opts), nil
	case arrow.UINT16:
		return unsignedProduct[uint16](opts), nil
	case arrow.UINT32:
		return unsignedProduct[uint32](opts), nil
	case arrow.UINT64:
		return unsignedProduct[uint64](opts), nil
	case arrow.FLOAT16:
		return floatProduct[float16.Num](opts, func(v float16.Num) float64 { return float64(v.Float32()) }), nil
	case arrow.FLOAT32:
		return floatProduct[float32](opts, func(v float32) float64 { return float64(v) }), nil
	case arrow.FLOAT64:
		return floatProduct[float64](opts, func(v float64) float64 { return v }), nil
	case arrow.DECIMAL128:
		t := dt.(*arrow.Decimal128Type)
		one := decimal128.FromI64(1).IncreaseScaleBy(t.Scale)
		return &productState{opts: opts, outType: dt,
			acc: one,
			mul: func(a, b any) any {
				return a.(decimal128.Num).Mul(b.(decimal128.Num)).ReduceScaleBy(t.Scale, true)
			},
			iter: func(s *exec.ArraySpan, fn func(any)) {
				fixedIter[decimal128.Num](s, func(v decimal128.Num) { fn(v) })
			},
			toScalar: func(a any) scalar.Scalar {
				return scalar.NewDecimal128Scalar(a.(decimal128.Num), dt)
			}}, nil
	case arrow.DECIMAL256:
		t := dt.(*arrow.Decimal256Type)
		one := decimal256.FromI64(1).IncreaseScaleBy(t.Scale)
		return &productState{opts: opts, outType: dt,
			acc: one,
			mul: func(a, b any) any {
				return a.(decimal256.Num).Mul(b.(decimal256.Num)).ReduceScaleBy(t.Scale, true)
			},
			iter: func(s *exec.ArraySpan, fn func(any)) {
				fixedIter[decimal256.Num](s, func(v decimal256.Num) { fn(v) })
			},
			toScalar: func(a any) scalar.Scalar {
				return scalar.NewDecimal256Scalar(a.(decimal256.Num), dt)
			}}, nil
	default:
		return nil, fmt.Errorf("%w: product not implemented for %s", arrow.ErrNotImplemented, dt)
	}
}

func signedProduct[T arrow.IntType](opts ScalarAggregateOptions) *productState {
	return &productState{opts: opts, outType: arrow.PrimitiveTypes.Int64,
		acc: int64(1), mul: func(a, b any) any { return a.(int64) * b.(int64) },
		iter:     func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(int64(v)) }) },
		toScalar: func(a any) scalar.Scalar { return scalar.NewInt64Scalar(a.(int64)) }}
}

func unsignedProduct[T arrow.UintType](opts ScalarAggregateOptions) *productState {
	return &productState{opts: opts, outType: arrow.PrimitiveTypes.Uint64,
		acc: uint64(1), mul: func(a, b any) any { return a.(uint64) * b.(uint64) },
		iter:     func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(uint64(v)) }) },
		toScalar: func(a any) scalar.Scalar { return scalar.NewUint64Scalar(a.(uint64)) }}
}

func floatProduct[T arrow.FixedWidthType](opts ScalarAggregateOptions, conv func(T) float64) *productState {
	return &productState{opts: opts, outType: arrow.PrimitiveTypes.Float64,
		acc: float64(1), mul: func(a, b any) any { return a.(float64) * b.(float64) },
		iter:     func(s *exec.ArraySpan, fn func(any)) { fixedIter[T](s, func(v T) { fn(conv(v)) }) },
		toScalar: func(a any) scalar.Scalar { return scalar.NewFloat64Scalar(a.(float64)) }}
}

func productKernels() []exec.ScalarAggregateKernel {
	var out []exec.ScalarAggregateKernel
	out = append(out, aggKernel(arrow.Null, exec.NewOutputType(arrow.PrimitiveTypes.Int64), initProduct, false))
	out = append(out, aggKernel(arrow.FixedWidthTypes.Boolean, exec.NewOutputType(arrow.PrimitiveTypes.Uint64), initProduct, false))
	for _, dt := range signedIntTypes {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Int64), initProduct, false))
	}
	for _, dt := range unsignedIntTypes {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Uint64), initProduct, false))
	}
	for _, dt := range []arrow.DataType{arrow.FixedWidthTypes.Float16, arrow.PrimitiveTypes.Float32, arrow.PrimitiveTypes.Float64} {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Float64), initProduct, false))
	}
	out = append(out, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL128), identityOutType, initProduct, false))
	out = append(out, aggKernelMatchedComputed(exec.SameTypeID(arrow.DECIMAL256), identityOutType, initProduct, false))
	return out
}

// ----------------------------------------------------------------------
// MinMax / Min / Max

type minMaxState struct {
	opts         ScalarAggregateOptions
	field        int // -1 struct, 0 min, 1 max
	effectiveMin uint32
	acc          orderedAccessor
	min, max     any
	seen         bool
	hasNulls     bool
	count        int64
}

func (s *minMaxState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.hasNulls = s.hasNulls || nulls > 0
	if v.IsArray() {
		s.acc.iter(&v.Array, func(x any) {
			if !s.seen {
				s.min, s.max, s.seen = x, x, true
				return
			}
			if s.acc.less(x, s.min) {
				s.min = x
			}
			if s.acc.less(s.max, x) {
				s.max = x
			}
		})
	}
	return nil
}

func (s *minMaxState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*minMaxState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for min_max", arrow.ErrInvalid)
	}
	s.count += o.count
	s.hasNulls = s.hasNulls || o.hasNulls
	if o.seen {
		if !s.seen {
			s.min, s.max, s.seen = o.min, o.max, true
		} else {
			if s.acc.less(o.min, s.min) {
				s.min = o.min
			}
			if s.acc.less(s.max, o.max) {
				s.max = o.max
			}
		}
	}
	return nil
}

func (s *minMaxState) nullOut() scalar.Scalar {
	nullScalar := scalar.MakeNullScalar(s.acc.childType)
	if s.field < 0 {
		sc, _ := scalar.NewStructScalarWithNames([]scalar.Scalar{nullScalar, nullScalar}, []string{"min", "max"})
		return sc
	}
	return nullScalar
}

func (s *minMaxState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if (!s.opts.SkipNulls && s.hasNulls) || s.count < int64(s.effectiveMin) || !s.seen {
		return s.nullOut(), nil
	}
	minSc := s.acc.toScalar(s.min)
	maxSc := s.acc.toScalar(s.max)
	switch s.field {
	case 0:
		return minSc, nil
	case 1:
		return maxSc, nil
	default:
		return scalar.NewStructScalarWithNames([]scalar.Scalar{minSc, maxSc}, []string{"min", "max"})
	}
}

func makeMinMaxInit(field int) exec.KernelInitFn {
	return func(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
		opts, err := parseScalarAggOptions(args.Options)
		if err != nil {
			return nil, err
		}
		acc, err := orderedAccessorFor(args.Inputs[0])
		if err != nil {
			return nil, err
		}
		return &minMaxState{opts: opts, field: field, effectiveMin: max(uint32(1), opts.MinCount), acc: acc}, nil
	}
}

func minMaxOutType(_ *exec.KernelCtx, types []arrow.DataType) (arrow.DataType, error) {
	t := types[0]
	return arrow.StructOf(
		arrow.Field{Name: "min", Type: t, Nullable: true},
		arrow.Field{Name: "max", Type: t, Nullable: true}), nil
}

func minMaxKernels() (mm, mn, mx []exec.ScalarAggregateKernel) {
	for _, dt := range orderedTypes() {
		mm = append(mm, aggKernelComputed(dt, minMaxOutType, makeMinMaxInit(-1), false))
		mn = append(mn, aggKernel(dt, exec.NewOutputType(dt), makeMinMaxInit(0), false))
		mx = append(mx, aggKernel(dt, exec.NewOutputType(dt), makeMinMaxInit(1), false))
	}
	mm = appendDecimalKernels(mm, minMaxOutType, makeMinMaxInit(-1), false)
	mn = appendDecimalKernels(mn, identityOutType, makeMinMaxInit(0), false)
	mx = appendDecimalKernels(mx, identityOutType, makeMinMaxInit(1), false)
	return
}

// ----------------------------------------------------------------------
// Any / All

type boolAggState struct {
	opts     ScalarAggregateOptions
	isAny    bool
	value    bool
	hasNulls bool
	count    int64
}

func (s *boolAggState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.hasNulls = s.hasNulls || nulls > 0
	if v.IsArray() {
		boolIter(&v.Array, func(b bool) {
			if s.isAny {
				s.value = s.value || b
			} else {
				s.value = s.value && b
			}
		})
	}
	return nil
}

func (s *boolAggState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*boolAggState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for boolean aggregate", arrow.ErrInvalid)
	}
	if s.isAny {
		s.value = s.value || o.value
	} else {
		s.value = s.value && o.value
	}
	s.hasNulls = s.hasNulls || o.hasNulls
	s.count += o.count
	return nil
}

func (s *boolAggState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	nullResult := false
	if s.isAny {
		nullResult = (!s.opts.SkipNulls && !s.value && s.hasNulls) || s.count < int64(s.opts.MinCount)
	} else {
		nullResult = (!s.opts.SkipNulls && s.value && s.hasNulls) || s.count < int64(s.opts.MinCount)
	}
	if nullResult {
		return scalar.MakeNullScalar(arrow.FixedWidthTypes.Boolean), nil
	}
	return scalar.NewBooleanScalar(s.value), nil
}

func initBoolAgg(isAny bool) exec.KernelInitFn {
	return func(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
		opts, err := parseScalarAggOptions(args.Options)
		if err != nil {
			return nil, err
		}
		return &boolAggState{opts: opts, isAny: isAny, value: !isAny}, nil
	}
}

func anyAllKernels(isAny bool) []exec.ScalarAggregateKernel {
	return []exec.ScalarAggregateKernel{
		aggKernel(arrow.FixedWidthTypes.Boolean, exec.NewOutputType(arrow.FixedWidthTypes.Boolean), initBoolAgg(isAny), false),
	}
}

// ----------------------------------------------------------------------
// First / Last

type firstLastState struct {
	opts         ScalarAggregateOptions
	field        int // -1 struct, 0 first, 1 last
	effectiveMin uint32
	acc          orderedAccessor
	count        int64
	hasAnyValues bool
	hasValues    bool
	first, last  any
	firstIsNull  bool
	lastIsNull   bool
}

func (s *firstLastState) mergeOne(v any) {
	if !s.hasValues {
		s.first = v
		s.hasValues = true
	}
	s.last = v
}

func (s *firstLastState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	if !v.IsArray() {
		return fmt.Errorf("%w: first/last requires array input", arrow.ErrInvalid)
	}
	arr := &v.Array
	nulls := valueNullCount(v, span.Len)
	s.count += span.Len - nulls
	s.hasAnyValues = true
	if span.Len == 0 {
		return nil
	}
	if nulls == 0 {
		s.mergeOne(s.acc.at(arr, 0))
		s.mergeOne(s.acc.at(arr, span.Len-1))
		return nil
	}
	if !s.hasValues && !validAt(arr, 0) {
		s.firstIsNull = true
	}
	if !validAt(arr, span.Len-1) {
		s.lastIsNull = true
	}
	for i := int64(0); i < span.Len; i++ {
		if validAt(arr, i) {
			s.mergeOne(s.acc.at(arr, i))
			break
		}
	}
	for i := span.Len - 1; i >= 0; i-- {
		if validAt(arr, i) {
			s.mergeOne(s.acc.at(arr, i))
			break
		}
	}
	return nil
}

func (s *firstLastState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*firstLastState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for first/last", arrow.ErrInvalid)
	}
	if !s.hasValues {
		s.first = o.first
	}
	if !s.hasAnyValues {
		s.firstIsNull = o.firstIsNull
	}
	if o.hasValues {
		s.last = o.last
	}
	s.lastIsNull = o.lastIsNull
	s.hasValues = s.hasValues || o.hasValues
	s.hasAnyValues = s.hasAnyValues || o.hasAnyValues
	s.count += o.count
	return nil
}

func (s *firstLastState) nullOut() scalar.Scalar {
	nullScalar := scalar.MakeNullScalar(s.acc.childType)
	if s.field < 0 {
		sc, _ := scalar.NewStructScalarWithNames([]scalar.Scalar{nullScalar, nullScalar}, []string{"first", "last"})
		return sc
	}
	return nullScalar
}

func (s *firstLastState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if s.count < int64(s.effectiveMin) {
		return s.nullOut(), nil
	}
	if !s.hasValues {
		return s.nullOut(), nil
	}
	nullScalar := scalar.MakeNullScalar(s.acc.childType)
	first := nullScalar
	last := nullScalar
	if s.opts.SkipNulls {
		first = s.acc.toScalar(s.first)
		last = s.acc.toScalar(s.last)
	} else {
		if !s.firstIsNull {
			first = s.acc.toScalar(s.first)
		}
		if !s.lastIsNull {
			last = s.acc.toScalar(s.last)
		}
	}
	switch s.field {
	case 0:
		return first, nil
	case 1:
		return last, nil
	default:
		return scalar.NewStructScalarWithNames([]scalar.Scalar{first, last}, []string{"first", "last"})
	}
}

func makeFirstLastInit(field int) exec.KernelInitFn {
	return func(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
		opts, err := parseScalarAggOptions(args.Options)
		if err != nil {
			return nil, err
		}
		acc, err := orderedAccessorFor(args.Inputs[0])
		if err != nil {
			return nil, err
		}
		return &firstLastState{opts: opts, field: field, effectiveMin: max(uint32(1), opts.MinCount), acc: acc}, nil
	}
}

func firstLastOutType(_ *exec.KernelCtx, types []arrow.DataType) (arrow.DataType, error) {
	t := types[0]
	return arrow.StructOf(
		arrow.Field{Name: "first", Type: t, Nullable: true},
		arrow.Field{Name: "last", Type: t, Nullable: true}), nil
}

func firstLastKernels() (fl, first, last []exec.ScalarAggregateKernel) {
	for _, dt := range orderedTypes() {
		fl = append(fl, aggKernelComputed(dt, firstLastOutType, makeFirstLastInit(-1), true))
		first = append(first, aggKernel(dt, exec.NewOutputType(dt), makeFirstLastInit(0), true))
		last = append(last, aggKernel(dt, exec.NewOutputType(dt), makeFirstLastInit(1), true))
	}
	fl = appendDecimalKernels(fl, firstLastOutType, makeFirstLastInit(-1), true)
	first = appendDecimalKernels(first, identityOutType, makeFirstLastInit(0), true)
	last = appendDecimalKernels(last, identityOutType, makeFirstLastInit(1), true)
	return
}

// ----------------------------------------------------------------------
// Index

// IndexOptions controls the index aggregate kernel.
type IndexOptions struct {
	// Value is the value to search for. It is required.
	Value scalar.Scalar `compute:"value"`
}

func (IndexOptions) TypeName() string { return "IndexOptions" }

type indexState struct {
	acc     orderedAccessor
	desired any
	seen    int64
	index   int64
	found   bool
}

func (s *indexState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if s.found || len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	if !v.IsArray() {
		return fmt.Errorf("%w: index requires array input", arrow.ErrInvalid)
	}
	arr := &v.Array
	base := s.seen
	for i := int64(0); i < span.Len; i++ {
		if validAt(arr, i) && s.acc.equal(s.acc.at(arr, i), s.desired) {
			s.index = base + i
			s.found = true
			break
		}
	}
	s.seen += span.Len
	return nil
}

func (s *indexState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*indexState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for index", arrow.ErrInvalid)
	}
	if !s.found && o.found {
		s.index = s.seen + o.index
		s.found = true
	}
	s.seen += o.seen
	return nil
}

func (s *indexState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if s.found {
		return scalar.NewInt64Scalar(s.index), nil
	}
	return scalar.NewInt64Scalar(-1), nil
}

func initIndex(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	var idxOpts IndexOptions
	switch v := args.Options.(type) {
	case IndexOptions:
		idxOpts = v
	case *IndexOptions:
		if v != nil {
			idxOpts = *v
		}
	}
	if idxOpts.Value == nil {
		return nil, fmt.Errorf("%w: must provide IndexOptions.value for index kernel", arrow.ErrInvalid)
	}
	acc, err := orderedAccessorFor(args.Inputs[0])
	if err != nil {
		return nil, err
	}
	return &indexState{acc: acc, desired: acc.fromScalar(idxOpts.Value), index: -1}, nil
}

func indexKernels() []exec.ScalarAggregateKernel {
	var out []exec.ScalarAggregateKernel
	for _, dt := range orderedTypes() {
		out = append(out, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Int64), initIndex, true))
	}
	out = append(out, aggKernelMatched(exec.SameTypeID(arrow.DECIMAL128), exec.NewOutputType(arrow.PrimitiveTypes.Int64), initIndex, true))
	out = append(out, aggKernelMatched(exec.SameTypeID(arrow.DECIMAL256), exec.NewOutputType(arrow.PrimitiveTypes.Int64), initIndex, true))
	return out
}

// ----------------------------------------------------------------------
// Variance / StdDev

// VarianceOptions controls the variance and stddev kernels.
type VarianceOptions struct {
	// DDOF is the delta degrees of freedom. The divisor is N - ddof.
	DDOF int `compute:"ddof"`
	// SkipNulls ignores null values (the default).
	SkipNulls bool `compute:"skip_nulls"`
	// MinCount is the minimum number of non-null values required.
	MinCount uint32 `compute:"min_count"`
}

func (VarianceOptions) TypeName() string { return "VarianceOptions" }

func DefaultVarianceOptions() VarianceOptions {
	return VarianceOptions{SkipNulls: true}
}

func parseVarianceOptions(opts any) (VarianceOptions, error) {
	switch value := opts.(type) {
	case nil:
		return DefaultVarianceOptions(), nil
	case VarianceOptions:
		return value, nil
	case *VarianceOptions:
		if value == nil {
			return DefaultVarianceOptions(), nil
		}
		return *value, nil
	default:
		return VarianceOptions{}, fmt.Errorf("%w: invalid VarianceOptions", arrow.ErrInvalid)
	}
}

type varianceState struct {
	opts   VarianceOptions
	isStd  bool
	count  int64
	mean   float64
	m2     float64
	nulls  bool
	iter   func(*exec.ArraySpan, func(float64))
	isNull bool
}

func (s *varianceState) Consume(_ *exec.KernelCtx, span *exec.ExecSpan) error {
	if len(span.Values) == 0 {
		return nil
	}
	v := &span.Values[0]
	nulls := valueNullCount(v, span.Len)
	s.nulls = s.nulls || nulls > 0
	if v.IsArray() {
		s.iter(&v.Array, func(x float64) {
			s.count++
			delta := x - s.mean
			s.mean += delta / float64(s.count)
			s.m2 += delta * (x - s.mean)
		})
	}
	return nil
}

func (s *varianceState) Merge(_ *exec.KernelCtx, src exec.KernelState) error {
	o, ok := src.(*varianceState)
	if !ok {
		return fmt.Errorf("%w: invalid source state for variance", arrow.ErrInvalid)
	}
	if o.count > 0 {
		if s.count == 0 {
			s.count, s.mean, s.m2 = o.count, o.mean, o.m2
		} else {
			total := s.count + o.count
			delta := o.mean - s.mean
			s.mean += delta * float64(o.count) / float64(total)
			s.m2 += o.m2 + delta*delta*float64(s.count)*float64(o.count)/float64(total)
			s.count = total
		}
	}
	s.nulls = s.nulls || o.nulls
	return nil
}

func (s *varianceState) Finalize(_ *exec.KernelCtx) (scalar.Scalar, error) {
	if s.isNull {
		if s.opts.SkipNulls && s.opts.MinCount == 0 && s.opts.DDOF == 0 {
			return scalar.NewFloat64Scalar(0), nil
		}
		return scalar.MakeNullScalar(arrow.PrimitiveTypes.Float64), nil
	}
	if s.count <= int64(s.opts.DDOF) || s.count < int64(s.opts.MinCount) || (!s.opts.SkipNulls && s.nulls) {
		return scalar.MakeNullScalar(arrow.PrimitiveTypes.Float64), nil
	}
	variance := s.m2 / float64(s.count-int64(s.opts.DDOF))
	if s.isStd {
		return scalar.NewFloat64Scalar(math.Sqrt(variance)), nil
	}
	return scalar.NewFloat64Scalar(variance), nil
}

func makeVarianceInit(isStd bool) exec.KernelInitFn {
	return func(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
		opts, err := parseVarianceOptions(args.Options)
		if err != nil {
			return nil, err
		}
		state := &varianceState{opts: opts, isStd: isStd}
		switch args.Inputs[0].ID() {
		case arrow.NULL:
			state.isNull = true
		case arrow.BOOL:
			state.iter = func(s *exec.ArraySpan, fn func(float64)) {
				boolIter(s, func(v bool) {
					if v {
						fn(1)
					} else {
						fn(0)
					}
				})
			}
		case arrow.INT8:
			state.iter = floatIterFn[int8](func(v int8) float64 { return float64(v) })
		case arrow.INT16:
			state.iter = floatIterFn[int16](func(v int16) float64 { return float64(v) })
		case arrow.INT32:
			state.iter = floatIterFn[int32](func(v int32) float64 { return float64(v) })
		case arrow.INT64:
			state.iter = floatIterFn[int64](func(v int64) float64 { return float64(v) })
		case arrow.UINT8:
			state.iter = floatIterFn[uint8](func(v uint8) float64 { return float64(v) })
		case arrow.UINT16:
			state.iter = floatIterFn[uint16](func(v uint16) float64 { return float64(v) })
		case arrow.UINT32:
			state.iter = floatIterFn[uint32](func(v uint32) float64 { return float64(v) })
		case arrow.UINT64:
			state.iter = floatIterFn[uint64](func(v uint64) float64 { return float64(v) })
		case arrow.FLOAT16:
			state.iter = floatIterFn[float16.Num](func(v float16.Num) float64 { return float64(v.Float32()) })
		case arrow.FLOAT32:
			state.iter = floatIterFn[float32](func(v float32) float64 { return float64(v) })
		case arrow.FLOAT64:
			state.iter = floatIterFn[float64](func(v float64) float64 { return v })
		default:
			return nil, fmt.Errorf("%w: variance not implemented for %s", arrow.ErrNotImplemented, args.Inputs[0])
		}
		return state, nil
	}
}

func floatIterFn[T arrow.FixedWidthType](conv func(T) float64) func(*exec.ArraySpan, func(float64)) {
	return func(s *exec.ArraySpan, fn func(float64)) {
		fixedIter[T](s, func(v T) { fn(conv(v)) })
	}
}

func varianceKernels() (variance, stddev []exec.ScalarAggregateKernel) {
	var types []arrow.DataType
	types = append(types, arrow.Null, arrow.FixedWidthTypes.Boolean)
	types = append(types, numericTypes...)
	for _, dt := range types {
		variance = append(variance, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Float64), makeVarianceInit(false), false))
		stddev = append(stddev, aggKernel(dt, exec.NewOutputType(arrow.PrimitiveTypes.Float64), makeVarianceInit(true), false))
	}
	return
}
