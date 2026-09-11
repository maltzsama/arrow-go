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
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHashSum(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 4]`)
	defer values.Release()
	keys := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "b", "a", "b"]`)
	defer keys.Release()

	res, err := compute.HashSum(ctx, compute.DefaultScalarAggregateOptions(),
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer res.Release()

	expected := aggArray(t, mem, arrow.PrimitiveTypes.Int64, `[4, 6]`)
	defer expected.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expected.Data()}, res, nil, nil)
}

func TestHashCountAndMean(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, null, 3, 4]`)
	defer values.Release()
	keys := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "a", "b", "b"]`)
	defer keys.Release()

	count, err := compute.HashCount(ctx, compute.CountOptions{},
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer count.Release()
	expectedCount := aggArray(t, mem, arrow.PrimitiveTypes.Int64, `[1, 2]`)
	defer expectedCount.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expectedCount.Data()}, count, nil, nil)

	mean, err := compute.HashMean(ctx, compute.DefaultScalarAggregateOptions(),
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer mean.Release()
	expectedMean := aggArray(t, mem, arrow.PrimitiveTypes.Float64, `[1.0, 3.5]`)
	defer expectedMean.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expectedMean.Data()}, mean, nil, nil)
}

func TestHashMinMax(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[3, 1, 5, 2]`)
	defer values.Release()
	keys := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "a", "b", "b"]`)
	defer keys.Release()

	mn, err := compute.HashMin(ctx, compute.DefaultScalarAggregateOptions(),
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer mn.Release()
	expectedMin := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2]`)
	defer expectedMin.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expectedMin.Data()}, mn, nil, nil)

	mx, err := compute.HashMax(ctx, compute.DefaultScalarAggregateOptions(),
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer mx.Release()
	expectedMax := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[3, 5]`)
	defer expectedMax.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expectedMax.Data()}, mx, nil, nil)
}

func TestHashCountDistinct(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 1, 2, 3, 3]`)
	defer values.Release()
	keys := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "a", "a", "b", "b"]`)
	defer keys.Release()

	res, err := compute.HashCountDistinct(ctx, compute.CountOptions{},
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer res.Release()
	expected := aggArray(t, mem, arrow.PrimitiveTypes.Int64, `[2, 1]`)
	defer expected.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expected.Data()}, res, nil, nil)
}

func TestHashMultiKey(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 4]`)
	defer values.Release()
	k1 := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "a", "b", "b"]`)
	defer k1.Release()
	k2 := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 1, 2]`)
	defer k2.Release()

	res, err := compute.HashSum(ctx, compute.DefaultScalarAggregateOptions(),
		&compute.ArrayDatum{Value: values.Data()},
		&compute.ArrayDatum{Value: k1.Data()},
		&compute.ArrayDatum{Value: k2.Data()})
	require.NoError(t, err)
	defer res.Release()
	expected := aggArray(t, mem, arrow.PrimitiveTypes.Int64, `[1, 2, 3, 4]`)
	defer expected.Release()
	assertDatumsEqual(t, &compute.ArrayDatum{Value: expected.Data()}, res, nil, nil)
}

func TestHashDistinctList(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 1, 2, 3]`)
	defer values.Release()
	keys := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "a", "b", "a"]`)
	defer keys.Release()

	distinct, err := compute.HashDistinct(ctx,
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer distinct.Release()
	assert.Equal(t, int64(2), distinct.(compute.ArrayLikeDatum).Len())

	list, err := compute.HashList(ctx,
		&compute.ArrayDatum{Value: values.Data()}, &compute.ArrayDatum{Value: keys.Data()})
	require.NoError(t, err)
	defer list.Release()
	assert.Equal(t, int64(2), list.(compute.ArrayLikeDatum).Len())
}
