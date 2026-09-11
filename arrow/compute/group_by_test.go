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
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGroupBy(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	keys := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "b", "a", "b"]`)
	defer keys.Release()
	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 2, 3, 4]`)
	defer values.Release()

	tbl, err := compute.GroupBy(ctx,
		[]compute.Datum{&compute.ArrayDatum{Value: keys.Data()}},
		[]compute.Aggregate{{
			Function: "sum",
			Options:  compute.DefaultScalarAggregateOptions(),
			Value:    &compute.ArrayDatum{Value: values.Data()},
		}})
	require.NoError(t, err)
	defer tbl.Release()

	assert.Equal(t, int64(2), tbl.NumRows())
	assert.Equal(t, int64(2), tbl.NumCols())
	assert.Equal(t, "key_0", tbl.Schema().Field(0).Name)
	assert.Equal(t, "sum", tbl.Schema().Field(1).Name)

	keyCol := tbl.Column(0).Data().Chunk(0)
	keyScalar, err := scalar.GetScalar(keyCol, 0)
	require.NoError(t, err)
	assert.True(t, scalar.Equals(scalar.NewStringScalar("a"), keyScalar))

	sumCol := tbl.Column(1).Data().Chunk(0)
	sumScalar, err := scalar.GetScalar(sumCol, 0)
	require.NoError(t, err)
	assert.True(t, scalar.Equals(scalar.NewInt64Scalar(4), sumScalar))
}

func TestGroupByMultiKeyMultiAgg(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	k1 := aggArray(t, mem, arrow.BinaryTypes.String, `["a", "a", "b", "b"]`)
	defer k1.Release()
	k2 := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[1, 1, 1, 2]`)
	defer k2.Release()
	values := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[10, 20, 30, 40]`)
	defer values.Release()

	tbl, err := compute.GroupBy(ctx,
		[]compute.Datum{
			&compute.ArrayDatum{Value: k1.Data()},
			&compute.ArrayDatum{Value: k2.Data()},
		},
		[]compute.Aggregate{
			{Function: "sum", Options: compute.DefaultScalarAggregateOptions(), Value: &compute.ArrayDatum{Value: values.Data()}},
			{Function: "count", Options: compute.CountOptions{}, Value: &compute.ArrayDatum{Value: values.Data()}},
		})
	require.NoError(t, err)
	defer tbl.Release()

	assert.Equal(t, int64(3), tbl.NumRows())
	assert.Equal(t, int64(4), tbl.NumCols())
	assert.Equal(t, "key_0", tbl.Schema().Field(0).Name)
	assert.Equal(t, "key_1", tbl.Schema().Field(1).Name)
	assert.Equal(t, "sum", tbl.Schema().Field(2).Name)
	assert.Equal(t, "count", tbl.Schema().Field(3).Name)
}
