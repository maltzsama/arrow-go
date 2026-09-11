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
	"math"
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

func aggArray(t *testing.T, mem memory.Allocator, typ arrow.DataType, values string) arrow.Array {
	t.Helper()
	arr, _, err := array.FromJSON(mem, typ, strings.NewReader(values))
	require.NoError(t, err)
	return arr
}

func aggScalar(t *testing.T, d compute.Datum) scalar.Scalar {
	t.Helper()
	sd, ok := d.(*compute.ScalarDatum)
	require.Truef(t, ok, "expected scalar datum, got %T", d)
	return sd.Value
}

func TestSumKernels(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	tests := []struct {
		name string
		typ  arrow.DataType
		in   string
		want scalar.Scalar
	}{
		{"int32", arrow.PrimitiveTypes.Int32, `[1, 2, 3]`, scalar.NewInt64Scalar(6)},
		{"int32_nulls", arrow.PrimitiveTypes.Int32, `[1, null, 3]`, scalar.NewInt64Scalar(4)},
		{"uint8", arrow.PrimitiveTypes.Uint8, `[1, 2, 3]`, scalar.NewUint64Scalar(6)},
		{"float64", arrow.PrimitiveTypes.Float64, `[1.5, 2.5]`, scalar.NewFloat64Scalar(4)},
		{"bool", arrow.FixedWidthTypes.Boolean, `[true, false, true]`, scalar.NewUint64Scalar(2)},
		{"empty", arrow.PrimitiveTypes.Int32, `[]`, scalar.NewInt64Scalar(0)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := aggArray(t, mem, tc.typ, tc.in)
			defer in.Release()
			res, err := compute.Sum(ctx, def, &compute.ArrayDatum{Value: in.Data()})
			require.NoError(t, err)
			defer res.Release()
			assert.Truef(t, scalar.Equals(tc.want, aggScalar(t, res)), "want %s got %s", tc.want, res)
		})
	}
}

func TestSumSkipNullsFalse(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, null, 3]`)
	defer in.Release()
	opts := compute.ScalarAggregateOptions{SkipNulls: false}
	res, err := compute.Sum(ctx, opts, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	assert.True(t, !aggScalar(t, res).IsValid())
}

func TestMeanKernels(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 4]`)
	defer in.Release()
	res, err := compute.Mean(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	assert.True(t, scalar.Equals(scalar.NewFloat64Scalar(2.5), aggScalar(t, res)))

	empty := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[]`)
	defer empty.Release()
	res2, err := compute.Mean(ctx, def, &compute.ArrayDatum{Value: empty.Data()})
	require.NoError(t, err)
	defer res2.Release()
	got := aggScalar(t, res2).(*scalar.Float64).Value
	assert.True(t, math.IsNaN(got), "expected NaN, got %v", got)
}

func TestProduct(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[2, 3, 4]`)
	defer in.Release()
	res, err := compute.Product(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(24), aggScalar(t, res)))
}

func TestMinMax(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	tests := []struct {
		name    string
		typ     arrow.DataType
		in      string
		wantMin scalar.Scalar
		wantMax scalar.Scalar
	}{
		{"int32", arrow.PrimitiveTypes.Int32, `[3, 1, 4, 1, 5]`, scalar.NewInt32Scalar(1), scalar.NewInt32Scalar(5)},
		{"int32_nulls", arrow.PrimitiveTypes.Int32, `[3, null, 1, 5]`, scalar.NewInt32Scalar(1), scalar.NewInt32Scalar(5)},
		{"float64", arrow.PrimitiveTypes.Float64, `[1.5, -2.5, 3.0]`, scalar.NewFloat64Scalar(-2.5), scalar.NewFloat64Scalar(3)},
		{"bool", arrow.FixedWidthTypes.Boolean, `[true, false, true]`, scalar.NewBooleanScalar(false), scalar.NewBooleanScalar(true)},
		{"string", arrow.BinaryTypes.String, `["banana", "apple", "cherry"]`, scalar.NewStringScalar("apple"), scalar.NewStringScalar("cherry")},
		{"date32", arrow.FixedWidthTypes.Date32, `[100, 50, 200]`, scalar.NewDate32Scalar(50), scalar.NewDate32Scalar(200)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := aggArray(t, mem, tc.typ, tc.in)
			defer in.Release()
			res, err := compute.MinMax(ctx, def, &compute.ArrayDatum{Value: in.Data()})
			require.NoError(t, err)
			defer res.Release()
			st := aggScalar(t, res).(*scalar.Struct)
			assert.Truef(t, scalar.Equals(tc.wantMin, st.Value[0]), "min want %s got %s", tc.wantMin, st.Value[0])
			assert.Truef(t, scalar.Equals(tc.wantMax, st.Value[1]), "max want %s got %s", tc.wantMax, st.Value[1])

			mn, err := compute.Min(ctx, def, &compute.ArrayDatum{Value: in.Data()})
			require.NoError(t, err)
			defer mn.Release()
			assert.Truef(t, scalar.Equals(tc.wantMin, aggScalar(t, mn)), "min want %s got %s", tc.wantMin, mn)

			mx, err := compute.Max(ctx, def, &compute.ArrayDatum{Value: in.Data()})
			require.NoError(t, err)
			defer mx.Release()
			assert.Truef(t, scalar.Equals(tc.wantMax, aggScalar(t, mx)), "max want %s got %s", tc.wantMax, mx)
		})
	}
}

func TestMinMaxAllNull(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[null, null]`)
	defer in.Release()
	res, err := compute.MinMax(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	st := aggScalar(t, res).(*scalar.Struct)
	assert.False(t, st.Value[0].IsValid())
	assert.False(t, st.Value[1].IsValid())
}

func TestAnyAll(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	tests := []struct {
		in      string
		wantAny bool
		wantAll bool
	}{
		{`[true, false, true]`, true, false},
		{`[true, true]`, true, true},
		{`[false, false]`, false, false},
		{`[true, null]`, true, true},
		{`[false, null]`, false, false},
	}
	for _, tc := range tests {
		in := aggArray(t, mem, arrow.FixedWidthTypes.Boolean, tc.in)
		resAny, err := compute.Any(ctx, def, &compute.ArrayDatum{Value: in.Data()})
		require.NoError(t, err)
		assert.Equal(t, tc.wantAny, aggScalar(t, resAny).(*scalar.Boolean).Value)
		resAny.Release()

		resAll, err := compute.All(ctx, def, &compute.ArrayDatum{Value: in.Data()})
		require.NoError(t, err)
		assert.Equal(t, tc.wantAll, aggScalar(t, resAll).(*scalar.Boolean).Value)
		resAll.Release()
		in.Release()
	}
}

func TestFirstLast(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[null, 2, 3, null]`)
	defer in.Release()
	res, err := compute.FirstLast(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	st := aggScalar(t, res).(*scalar.Struct)
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(2), st.Value[0]))
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(3), st.Value[1]))

	first, err := compute.First(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer first.Release()
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(2), aggScalar(t, first)))

	last, err := compute.Last(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer last.Release()
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(3), aggScalar(t, last)))
}

func TestCountDistinct(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 2, 3, null, 1]`)
	defer in.Release()
	res, err := compute.CountDistinct(ctx, compute.CountOptions{}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(3), aggScalar(t, res)))

	resAll, err := compute.CountDistinct(ctx, compute.CountOptions{Mode: compute.CountAll}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer resAll.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(4), aggScalar(t, resAll)))

	strs := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "b", "a"]`)
	defer strs.Release()
	resStr, err := compute.CountDistinct(ctx, compute.CountOptions{}, &compute.ArrayDatum{Value: strs.Data()})
	require.NoError(t, err)
	defer resStr.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(2), aggScalar(t, resStr)))
}

func TestIndex(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 2]`)
	defer in.Release()
	res, err := compute.Index(ctx, compute.IndexOptions{Value: scalar.NewInt32Scalar(2)}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(1), aggScalar(t, res)))

	notFound, err := compute.Index(ctx, compute.IndexOptions{Value: scalar.NewInt32Scalar(9)}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer notFound.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(-1), aggScalar(t, notFound)))
}

func TestVarianceStdDev(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultVarianceOptions()

	in := aggArray(t, mem, arrow.PrimitiveTypes.Float64, `[2, 4, 4, 4, 5, 5, 7, 9]`)
	defer in.Release()
	res, err := compute.Variance(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	assert.InDelta(t, 4.0, aggScalar(t, res).(*scalar.Float64).Value, 1e-9)

	std, err := compute.StdDev(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer std.Release()
	assert.InDelta(t, 2.0, aggScalar(t, std).(*scalar.Float64).Value, 1e-9)
}

func TestScalarAggregateChunked(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	c0 := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2]`)
	defer c0.Release()
	c1 := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[3, 4]`)
	defer c1.Release()
	chunked := arrow.NewChunked(arrow.PrimitiveTypes.Int32, []arrow.Array{c0, c1})
	defer chunked.Release()

	res, err := compute.Sum(ctx, def, &compute.ChunkedDatum{Value: chunked})
	require.NoError(t, err)
	defer res.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(10), aggScalar(t, res)))

	resMM, err := compute.MinMax(ctx, def, &compute.ChunkedDatum{Value: chunked})
	require.NoError(t, err)
	defer resMM.Release()
	st := aggScalar(t, resMM).(*scalar.Struct)
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(1), st.Value[0]))
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(4), st.Value[1]))
}

func TestDayTimeIntervalAgg(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	in := aggArray(t, mem, arrow.FixedWidthTypes.DayTimeInterval,
		`[{"days":3,"milliseconds":100}, {"days":1,"milliseconds":500}, {"days":2,"milliseconds":200}]`)
	defer in.Release()

	wantMin := scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 1, Milliseconds: 500})
	wantMax := scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 3, Milliseconds: 100})

	res, err := compute.MinMax(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	st := aggScalar(t, res).(*scalar.Struct)
	assert.True(t, scalar.Equals(wantMin, st.Value[0]))
	assert.True(t, scalar.Equals(wantMax, st.Value[1]))

	mn, err := compute.Min(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mn.Release()
	assert.True(t, scalar.Equals(wantMin, aggScalar(t, mn)))

	mx, err := compute.Max(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mx.Release()
	assert.True(t, scalar.Equals(wantMax, aggScalar(t, mx)))

	fl, err := compute.FirstLast(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer fl.Release()
	fst := aggScalar(t, fl).(*scalar.Struct)
	assert.True(t, scalar.Equals(scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 3, Milliseconds: 100}), fst.Value[0]))
	assert.True(t, scalar.Equals(scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 2, Milliseconds: 200}), fst.Value[1]))

	first, err := compute.First(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer first.Release()
	assert.True(t, scalar.Equals(scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 3, Milliseconds: 100}), aggScalar(t, first)))

	last, err := compute.Last(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer last.Release()
	assert.True(t, scalar.Equals(scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 2, Milliseconds: 200}), aggScalar(t, last)))

	idx, err := compute.Index(ctx, compute.IndexOptions{Value: scalar.NewDayTimeIntervalScalar(arrow.DayTimeInterval{Days: 1, Milliseconds: 500})}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer idx.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(1), aggScalar(t, idx)))
}

func TestMonthDayNanoIntervalAgg(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	in := aggArray(t, mem, arrow.FixedWidthTypes.MonthDayNanoInterval,
		`[{"months":3,"days":1,"nanoseconds":100}, {"months":1,"days":5,"nanoseconds":500}, {"months":2,"days":3,"nanoseconds":200}]`)
	defer in.Release()

	wantMin := scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 1, Days: 5, Nanoseconds: 500})
	wantMax := scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 3, Days: 1, Nanoseconds: 100})

	res, err := compute.MinMax(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	st := aggScalar(t, res).(*scalar.Struct)
	assert.True(t, scalar.Equals(wantMin, st.Value[0]))
	assert.True(t, scalar.Equals(wantMax, st.Value[1]))

	mn, err := compute.Min(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mn.Release()
	assert.True(t, scalar.Equals(wantMin, aggScalar(t, mn)))

	mx, err := compute.Max(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mx.Release()
	assert.True(t, scalar.Equals(wantMax, aggScalar(t, mx)))

	fl, err := compute.FirstLast(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer fl.Release()
	fst := aggScalar(t, fl).(*scalar.Struct)
	assert.True(t, scalar.Equals(scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 3, Days: 1, Nanoseconds: 100}), fst.Value[0]))
	assert.True(t, scalar.Equals(scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 2, Days: 3, Nanoseconds: 200}), fst.Value[1]))

	first, err := compute.First(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer first.Release()
	assert.True(t, scalar.Equals(scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 3, Days: 1, Nanoseconds: 100}), aggScalar(t, first)))

	last, err := compute.Last(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer last.Release()
	assert.True(t, scalar.Equals(scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 2, Days: 3, Nanoseconds: 200}), aggScalar(t, last)))

	idx, err := compute.Index(ctx, compute.IndexOptions{Value: scalar.NewMonthDayNanoIntervalScalar(arrow.MonthDayNanoInterval{Months: 1, Days: 5, Nanoseconds: 500})}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer idx.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(1), aggScalar(t, idx)))
}

func TestFixedSizeBinaryAgg(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	// 2-byte fixed-size binary: [1,3], [1,2], [1,0]
	fbType := &arrow.FixedSizeBinaryType{ByteWidth: 2}
	bldr := array.NewFixedSizeBinaryBuilder(mem, fbType)
	defer bldr.Release()
	bldr.Append([]byte{1, 3})
	bldr.Append([]byte{1, 2})
	bldr.Append([]byte{1, 0})
	in := bldr.NewArray()
	defer in.Release()

	wantMin := scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 0}), fbType)
	wantMax := scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 3}), fbType)

	res, err := compute.MinMax(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	st := aggScalar(t, res).(*scalar.Struct)
	assert.True(t, scalar.Equals(wantMin, st.Value[0]))
	assert.True(t, scalar.Equals(wantMax, st.Value[1]))

	mn, err := compute.Min(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mn.Release()
	assert.True(t, scalar.Equals(wantMin, aggScalar(t, mn)))

	mx, err := compute.Max(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mx.Release()
	assert.True(t, scalar.Equals(wantMax, aggScalar(t, mx)))

	fl, err := compute.FirstLast(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer fl.Release()
	fst := aggScalar(t, fl).(*scalar.Struct)
	assert.True(t, scalar.Equals(scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 3}), fbType), fst.Value[0]))
	assert.True(t, scalar.Equals(scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 0}), fbType), fst.Value[1]))

	first, err := compute.First(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer first.Release()
	assert.True(t, scalar.Equals(scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 3}), fbType), aggScalar(t, first)))

	last, err := compute.Last(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer last.Release()
	assert.True(t, scalar.Equals(scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 0}), fbType), aggScalar(t, last)))

	idx, err := compute.Index(ctx, compute.IndexOptions{Value: scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte{1, 2}), fbType)}, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer idx.Release()
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(1), aggScalar(t, idx)))
}
