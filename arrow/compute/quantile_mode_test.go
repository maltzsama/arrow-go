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
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQuantile(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 4]`)
	defer in.Release()

	res, err := compute.Quantile(ctx, compute.QuantileOptions{Q: []float64{0.5}, SkipNulls: true},
		&compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	expected := aggArray(t, mem, arrow.PrimitiveTypes.Float64, `[2.5]`)
	defer expected.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expected.Data()}, res, nil, nil)

	multi, err := compute.Quantile(ctx, compute.QuantileOptions{Q: []float64{0, 0.5, 1}, SkipNulls: true},
		&compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer multi.Release()
	expectedMulti := aggArray(t, mem, arrow.PrimitiveTypes.Float64, `[1, 2.5, 4]`)
	defer expectedMulti.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expectedMulti.Data()}, multi, nil, nil)
}

func TestApproximateMedian(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 4, 5]`)
	defer in.Release()
	res, err := compute.ApproximateMedian(ctx, compute.DefaultScalarAggregateOptions(),
		&compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	expected := aggArray(t, mem, arrow.PrimitiveTypes.Float64, `[3]`)
	defer expected.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expected.Data()}, res, nil, nil)
}

func TestMode(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 1, 2, 3]`)
	defer in.Release()
	res, err := compute.Mode(ctx, compute.DefaultModeOptions(), &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()

	arr := res.(*compute.ArrayDatum).MakeArray()
	defer arr.Release()
	st, ok := arr.(*array.Struct)
	require.True(t, ok)

	modeScalar, err := scalar.GetScalar(st.Field(0), 0)
	require.NoError(t, err)
	assert.True(t, scalar.Equals(scalar.NewInt32Scalar(1), modeScalar))

	countScalar, err := scalar.GetScalar(st.Field(1), 0)
	require.NoError(t, err)
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(2), countScalar))
}
