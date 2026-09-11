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
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func assertDecimal(t *testing.T, sc scalar.Scalar, want decimal128.Num, scale int32) {
	t.Helper()
	got, ok := sc.(*scalar.Decimal128)
	require.Truef(t, ok, "expected decimal128 scalar, got %T", sc)
	assert.Zerof(t, got.Value.Cmp(want), "got %s want %s", got.Value.ToString(scale), want.ToString(scale))
}

func TestDecimalSumMeanProduct(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	dt := &arrow.Decimal128Type{Precision: 10, Scale: 2}
	in := aggArray(t, mem, dt, `[1.23, 4.56]`)
	defer in.Release()

	sum, err := compute.Sum(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer sum.Release()
	assertDecimal(t, aggScalar(t, sum), decimal128.FromI64(579), 2)
	assert.Equal(t, int32(38), aggScalar(t, sum).DataType().(*arrow.Decimal128Type).Precision)

	mean, err := compute.Mean(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mean.Release()
	assertDecimal(t, aggScalar(t, mean), decimal128.FromI64(290), 2)

	prod, err := compute.Product(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer prod.Release()
	wantProd, err := decimal128.FromString("5.61", 10, 2)
	require.NoError(t, err)
	assertDecimal(t, aggScalar(t, prod), wantProd, 2)
}

func TestDecimalMinMaxFirstLastIndex(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)
	def := compute.DefaultScalarAggregateOptions()

	dt := &arrow.Decimal128Type{Precision: 10, Scale: 2}
	in := aggArray(t, mem, dt, `[3.00, 1.00, 5.00, 2.00]`)
	defer in.Release()

	mm, err := compute.MinMax(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mm.Release()
	st := aggScalar(t, mm).(*scalar.Struct)
	assertDecimal(t, st.Value[0], decimal128.FromI64(100), 2)
	assertDecimal(t, st.Value[1], decimal128.FromI64(500), 2)

	mn, err := compute.Min(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer mn.Release()
	assertDecimal(t, aggScalar(t, mn), decimal128.FromI64(100), 2)

	first, err := compute.First(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer first.Release()
	assertDecimal(t, aggScalar(t, first), decimal128.FromI64(300), 2)

	last, err := compute.Last(ctx, def, &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer last.Release()
	assertDecimal(t, aggScalar(t, last), decimal128.FromI64(200), 2)

	idx, err := compute.Index(ctx, compute.IndexOptions{Value: scalar.NewDecimal128Scalar(decimal128.FromI64(500), dt)},
		&compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer idx.Release()
	assert.Equal(t, int64(2), aggScalar(t, idx).(*scalar.Int64).Value)
}
