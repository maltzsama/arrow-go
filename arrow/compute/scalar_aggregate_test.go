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

package compute_test

import (
	"context"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func countInput(t *testing.T, mem memory.Allocator, typ arrow.DataType, values string) arrow.Array {
	t.Helper()
	arr, _, err := array.FromJSON(mem, typ, strings.NewReader(values))
	require.NoError(t, err)
	return arr
}

func TestCountArray(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	tests := []struct {
		name string
		typ  arrow.DataType
		in   string
		mode compute.CountMode
		want int64
	}{
		{name: "only_valid", typ: arrow.PrimitiveTypes.Int32, in: `[1, 2, null, 4]`, mode: compute.CountOnlyValid, want: 3},
		{name: "only_null", typ: arrow.PrimitiveTypes.Int32, in: `[1, 2, null, 4]`, mode: compute.CountOnlyNull, want: 1},
		{name: "all", typ: arrow.PrimitiveTypes.Int32, in: `[1, 2, null, 4]`, mode: compute.CountAll, want: 4},
		{name: "no_nulls", typ: arrow.PrimitiveTypes.Int32, in: `[1, 2, 3, 4]`, mode: compute.CountOnlyNull, want: 0},
		{name: "all_null", typ: arrow.PrimitiveTypes.Int32, in: `[null, null]`, mode: compute.CountOnlyValid, want: 0},
		{name: "all_null_mode_all", typ: arrow.PrimitiveTypes.Int32, in: `[null, null]`, mode: compute.CountAll, want: 2},
		{name: "empty", typ: arrow.PrimitiveTypes.Int32, in: `[]`, mode: compute.CountAll, want: 0},
		{name: "strings", typ: arrow.BinaryTypes.String, in: `["a", null, "c"]`, mode: compute.CountOnlyValid, want: 2},
		{name: "booleans", typ: arrow.FixedWidthTypes.Boolean, in: `[true, false, null]`, mode: compute.CountOnlyValid, want: 2},
		{name: "float64", typ: arrow.PrimitiveTypes.Float64, in: `[1.5, null, 3.5]`, mode: compute.CountOnlyNull, want: 1},
		{name: "uint8", typ: arrow.PrimitiveTypes.Uint8, in: `[1, 2, 3]`, mode: compute.CountAll, want: 3},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := countInput(t, mem, tc.typ, tc.in)
			defer input.Release()

			result, err := compute.Count(ctx, compute.CountOptions{Mode: tc.mode}, &compute.ArrayDatum{Value: input.Data()})
			require.NoError(t, err)
			defer result.Release()

			assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(tc.want)}, result, nil, nil)
		})
	}
}

func TestCountDefaultOptions(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	input := countInput(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, null, 4]`)
	defer input.Release()

	result, err := compute.CallFunction(ctx, "count", nil, &compute.ArrayDatum{Value: input.Data()})
	require.NoError(t, err)
	defer result.Release()

	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(3)}, result, nil, nil)
}

func TestCountNullArray(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	input := array.NewNull(5)
	defer input.Release()

	result, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountAll}, &compute.ArrayDatum{Value: input.Data()})
	require.NoError(t, err)
	defer result.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(5)}, result, nil, nil)

	valid, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountOnlyValid}, &compute.ArrayDatum{Value: input.Data()})
	require.NoError(t, err)
	defer valid.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(0)}, valid, nil, nil)
}

func TestCountChunked(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	chunk0 := countInput(t, mem, arrow.PrimitiveTypes.Int32, `[1, null, 3]`)
	defer chunk0.Release()
	chunk1 := countInput(t, mem, arrow.PrimitiveTypes.Int32, `[4, 5, null, null]`)
	defer chunk1.Release()

	chunked := arrow.NewChunked(arrow.PrimitiveTypes.Int32, []arrow.Array{chunk0, chunk1})
	defer chunked.Release()

	result, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountOnlyValid}, &compute.ChunkedDatum{Value: chunked})
	require.NoError(t, err)
	defer result.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(4)}, result, nil, nil)

	resultNull, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountOnlyNull}, &compute.ChunkedDatum{Value: chunked})
	require.NoError(t, err)
	defer resultNull.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(3)}, resultNull, nil, nil)
}

func TestCountScalar(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	valid := &compute.ScalarDatum{Value: scalar.NewInt64Scalar(42)}
	result, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountOnlyValid}, valid)
	require.NoError(t, err)
	defer result.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(1)}, result, nil, nil)

	nullScalar := &compute.ScalarDatum{Value: scalar.MakeNullScalar(arrow.PrimitiveTypes.Int64)}
	nullResult, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountOnlyNull}, nullScalar)
	require.NoError(t, err)
	defer nullResult.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(1)}, nullResult, nil, nil)

	allResult, err := compute.Count(ctx, compute.CountOptions{Mode: compute.CountAll}, nullScalar)
	require.NoError(t, err)
	defer allResult.Release()
	assertDatumsEqual(t, &compute.ScalarDatum{Value: scalar.NewInt64Scalar(1)}, allResult, nil, nil)
}

func TestCountResultType(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	input := countInput(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3]`)
	defer input.Release()

	result, err := compute.Count(ctx, compute.CountOptions{}, &compute.ArrayDatum{Value: input.Data()})
	require.NoError(t, err)
	defer result.Release()

	assert.Equal(t, arrow.PrimitiveTypes.Int64, result.(compute.ArrayLikeDatum).Type())
}
