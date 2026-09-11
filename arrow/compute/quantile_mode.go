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

package compute

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// Interpolation is the interpolation method used by Quantile when the quantile
// lies between two data points.
type Interpolation int8

const (
	// InterpolationLinear linearly interpolates between the two data points.
	InterpolationLinear Interpolation = iota
	// InterpolationLower returns the lower of the two data points.
	InterpolationLower
	// InterpolationHigher returns the higher of the two data points.
	InterpolationHigher
	// InterpolationNearest returns the nearest data point.
	InterpolationNearest
	// InterpolationMidpoint returns the midpoint of the two data points.
	InterpolationMidpoint
)

// QuantileOptions controls the quantile kernel.
type QuantileOptions struct {
	// Q is the list of quantiles to compute. Each value must be in [0, 1].
	// If empty, 0.5 (the median) is used.
	Q []float64 `compute:"q"`
	// Interpolation selects how values between data points are computed.
	Interpolation Interpolation `compute:"interpolation"`
	// SkipNulls ignores null values (the default).
	SkipNulls bool `compute:"skip_nulls"`
	// MinCount is the minimum number of non-null values required.
	MinCount uint32 `compute:"min_count"`
}

func (QuantileOptions) TypeName() string { return "QuantileOptions" }

// DefaultQuantileOptions returns the default quantile options (median,
// linear interpolation, skip nulls).
func DefaultQuantileOptions() QuantileOptions {
	return QuantileOptions{Q: []float64{0.5}, SkipNulls: true}
}

// ModeOptions controls the mode kernel.
type ModeOptions struct {
	// N is the number of modes to return. Defaults to 1.
	N int64 `compute:"n"`
	// SkipNulls ignores null values (the default).
	SkipNulls bool `compute:"skip_nulls"`
	// MinCount is the minimum number of non-null values required.
	MinCount uint32 `compute:"min_count"`
}

func (ModeOptions) TypeName() string { return "ModeOptions" }

// DefaultModeOptions returns the default mode options (single mode, skip
// nulls).
func DefaultModeOptions() ModeOptions {
	return ModeOptions{N: 1, SkipNulls: true}
}

func scalarToFloat64(sc scalar.Scalar) (float64, bool) {
	switch v := sc.(type) {
	case *scalar.Int8:
		return float64(v.Value), true
	case *scalar.Int16:
		return float64(v.Value), true
	case *scalar.Int32:
		return float64(v.Value), true
	case *scalar.Int64:
		return float64(v.Value), true
	case *scalar.Uint8:
		return float64(v.Value), true
	case *scalar.Uint16:
		return float64(v.Value), true
	case *scalar.Uint32:
		return float64(v.Value), true
	case *scalar.Uint64:
		return float64(v.Value), true
	case *scalar.Float16:
		return float64(v.Value.Float32()), true
	case *scalar.Float32:
		return float64(v.Value), true
	case *scalar.Float64:
		return float64(v.Value), true
	case *scalar.Decimal128:
		return v.Value.ToFloat64(v.DataType().(*arrow.Decimal128Type).Scale), true
	case *scalar.Decimal256:
		return v.Value.ToFloat64(v.DataType().(*arrow.Decimal256Type).Scale), true
	default:
		return 0, false
	}
}

// Quantile computes the requested quantiles of a numeric array. It returns an
// array with one element per requested quantile, or nulls when there is not
// enough valid data.
func Quantile(ctx context.Context, opts QuantileOptions, values Datum) (Datum, error) {
	arr, err := datumToArray(ctx, values)
	if err != nil {
		return nil, err
	}
	defer arr.Release()

	qs := opts.Q
	if len(qs) == 0 {
		qs = []float64{0.5}
	}
	for _, q := range qs {
		if q < 0 || q > 1 || math.IsNaN(q) {
			return nil, fmt.Errorf("%w: quantile must be between 0 and 1 inclusive", arrow.ErrInvalid)
		}
	}

	type point struct {
		f  float64
		sc scalar.Scalar
	}
	var points []point
	var hasNulls bool
	for i := 0; i < arr.Len(); i++ {
		sc, err := scalar.GetScalar(arr, i)
		if err != nil {
			return nil, err
		}
		if !sc.IsValid() {
			hasNulls = true
			continue
		}
		f, ok := scalarToFloat64(sc)
		if !ok {
			return nil, fmt.Errorf("%w: quantile requires a numeric input, got %s", arrow.ErrType, arr.DataType())
		}
		if math.IsNaN(f) {
			continue
		}
		points = append(points, point{f: f, sc: sc})
	}

	mem := exec.GetAllocator(ctx)
	dataPointInterp := opts.Interpolation == InterpolationLower ||
		opts.Interpolation == InterpolationHigher ||
		opts.Interpolation == InterpolationNearest

	var bldr array.Builder
	if dataPointInterp {
		bldr = array.NewBuilder(mem, arr.DataType())
	} else {
		bldr = array.NewBuilder(mem, arrow.PrimitiveTypes.Float64)
	}
	defer bldr.Release()

	nullResult := (!opts.SkipNulls && hasNulls) || len(points) < int(opts.MinCount)
	if !nullResult {
		slices.SortStableFunc(points, func(a, b point) int { return cmp.Compare(a.f, b.f) })
	}
	for _, q := range qs {
		if nullResult || len(points) == 0 {
			bldr.AppendNull()
			continue
		}
		pos := q * float64(len(points)-1)
		lo := int(math.Floor(pos))
		hi := int(math.Ceil(pos))
		frac := pos - float64(lo)
		var out scalar.Scalar
		switch opts.Interpolation {
		case InterpolationLower:
			out = points[lo].sc
		case InterpolationHigher:
			out = points[hi].sc
		case InterpolationNearest:
			idx := int(math.Round(pos))
			if idx < 0 {
				idx = 0
			}
			if idx >= len(points) {
				idx = len(points) - 1
			}
			out = points[idx].sc
		case InterpolationMidpoint:
			out = scalar.NewFloat64Scalar((points[lo].f + points[hi].f) / 2)
		default: // linear
			out = scalar.NewFloat64Scalar(points[lo].f + frac*(points[hi].f-points[lo].f))
		}
		if err := scalar.Append(bldr, out); err != nil {
			return nil, err
		}
	}
	out := bldr.NewArray()
	defer out.Release()
	return NewDatum(out), nil
}

// Mode computes the most common value(s) of an array and returns a struct
// array with fields "mode" and "count", ordered by descending count.
func Mode(ctx context.Context, opts ModeOptions, values Datum) (Datum, error) {
	arr, err := datumToArray(ctx, values)
	if err != nil {
		return nil, err
	}
	defer arr.Release()

	n := opts.N
	if n <= 0 {
		n = 1
	}

	type entry struct {
		sc    scalar.Scalar
		count int64
	}
	counts := make(map[string]*entry, arr.Len())
	var order []string
	var hasNulls bool
	var validCount int64
	for i := 0; i < arr.Len(); i++ {
		sc, err := scalar.GetScalar(arr, i)
		if err != nil {
			return nil, err
		}
		if !sc.IsValid() {
			hasNulls = true
			continue
		}
		validCount++
		key := string(encodeGroupKey(nil, sc))
		if e, ok := counts[key]; ok {
			e.count++
		} else {
			counts[key] = &entry{sc: sc, count: 1}
			order = append(order, key)
		}
	}

	var entries []*entry
	if !((!opts.SkipNulls && hasNulls) || validCount < int64(opts.MinCount)) {
		for _, key := range order {
			entries = append(entries, counts[key])
		}
		slices.SortStableFunc(entries, func(a, b *entry) int {
			return cmp.Compare(b.count, a.count)
		})
		if int64(len(entries)) > n {
			entries = entries[:n]
		}
	}

	mem := exec.GetAllocator(ctx)
	structType := arrow.StructOf(
		arrow.Field{Name: "mode", Type: arr.DataType(), Nullable: true},
		arrow.Field{Name: "count", Type: arrow.PrimitiveTypes.Int64, Nullable: true})
	sb := array.NewStructBuilder(mem, structType)
	defer sb.Release()

	for _, e := range entries {
		if err := scalar.Append(sb.FieldBuilder(0), e.sc); err != nil {
			return nil, err
		}
		if err := scalar.Append(sb.FieldBuilder(1), scalar.NewInt64Scalar(e.count)); err != nil {
			return nil, err
		}
		sb.Append(true)
	}
	out := sb.NewArray()
	defer out.Release()
	return NewDatum(out), nil
}
