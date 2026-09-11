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
	"context"
	"fmt"
	"math"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/internal/tdigest"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// TDigestOptions controls the tdigest kernel.
type TDigestOptions struct {
	// Q is the list of quantiles to compute. Each value must be in [0, 1].
	// If empty, 0.5 (the median) is used.
	Q []float64 `compute:"q"`
	// Delta is the compression parameter. Defaults to 100.
	Delta uint32 `compute:"delta"`
	// BufferSize is the input buffer size. Defaults to 500.
	BufferSize uint32 `compute:"buffer_size"`
	// SkipNulls ignores null values (the default).
	SkipNulls bool `compute:"skip_nulls"`
	// MinCount is the minimum number of non-null values required.
	MinCount uint32 `compute:"min_count"`
}

func (TDigestOptions) TypeName() string { return "TDigestOptions" }

// DefaultTDigestOptions returns the default tdigest options (median, delta
// 100, buffer size 500, skip nulls).
func DefaultTDigestOptions() TDigestOptions {
	return TDigestOptions{Q: []float64{0.5}, Delta: 100, BufferSize: 500, SkipNulls: true}
}

// buildTDigest consumes the valid numeric values of arr into a new TDigest.
// It returns the digest, the number of valid values, and whether any null was
// observed.
func buildTDigest(arr arrow.Array, delta, bufferSize uint32) (*tdigest.TDigest, int64, bool, error) {
	td := tdigest.New(delta, bufferSize)
	var count int64
	var hasNulls bool
	for i := 0; i < arr.Len(); i++ {
		sc, err := scalar.GetScalar(arr, i)
		if err != nil {
			return nil, 0, false, err
		}
		if !sc.IsValid() {
			hasNulls = true
			continue
		}
		f, ok := scalarToFloat64(sc)
		if !ok {
			return nil, 0, false, fmt.Errorf("%w: tdigest requires a numeric input, got %s", arrow.ErrType, arr.DataType())
		}
		if math.IsNaN(f) {
			continue
		}
		td.Add(f)
		count++
	}
	return td, count, hasNulls, nil
}

// TDigest computes approximate quantiles of a numeric array using the
// t-digest algorithm. It returns a float64 array with one element per
// requested quantile, or nulls when there is not enough valid data.
func TDigest(ctx context.Context, opts TDigestOptions, values Datum) (Datum, error) {
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
	delta := opts.Delta
	if delta == 0 {
		delta = 100
	}
	bufferSize := opts.BufferSize
	if bufferSize == 0 {
		bufferSize = 500
	}

	td, count, hasNulls, err := buildTDigest(arr, delta, bufferSize)
	if err != nil {
		return nil, err
	}

	mem := exec.GetAllocator(ctx)
	bldr := array.NewBuilder(mem, arrow.PrimitiveTypes.Float64)
	defer bldr.Release()

	nullResult := td.IsEmpty() || (!opts.SkipNulls && hasNulls) || count < int64(opts.MinCount)
	for _, q := range qs {
		if nullResult {
			bldr.AppendNull()
			continue
		}
		if err := scalar.Append(bldr, scalar.NewFloat64Scalar(td.Quantile(q))); err != nil {
			return nil, err
		}
	}
	out := bldr.NewArray()
	defer out.Release()
	return NewDatum(out), nil
}

// ApproximateMedian computes the approximate median (0.5 quantile) of a
// numeric array using the t-digest algorithm. It returns a float64 scalar, or
// a null scalar when there is not enough valid data.
func ApproximateMedian(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	arr, err := datumToArray(ctx, values)
	if err != nil {
		return nil, err
	}
	defer arr.Release()

	td, count, hasNulls, err := buildTDigest(arr, 100, 500)
	if err != nil {
		return nil, err
	}
	if td.IsEmpty() || (!opts.SkipNulls && hasNulls) || count < int64(opts.MinCount) {
		return NewDatum(scalar.MakeNullScalar(arrow.PrimitiveTypes.Float64)), nil
	}
	return NewDatum(scalar.NewFloat64Scalar(td.Quantile(0.5))), nil
}
