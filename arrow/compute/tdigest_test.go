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
	"fmt"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTDigest(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	var sb strings.Builder
	sb.WriteByte('[')
	for i := 1; i <= 1000; i++ {
		if i > 1 {
			sb.WriteByte(',')
		}
		fmt.Fprintf(&sb, "%d", i)
	}
	sb.WriteByte(']')
	in, _, err := array.FromJSON(mem, arrow.PrimitiveTypes.Int32, strings.NewReader(sb.String()))
	require.NoError(t, err)
	defer in.Release()

	res, err := compute.TDigest(ctx, compute.TDigestOptions{Q: []float64{0.5}, SkipNulls: true},
		&compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()

	arr := res.(*compute.ArrayDatum).MakeArray()
	defer arr.Release()
	assert.Equal(t, 1, arr.Len())
	median := arr.(*array.Float64).Value(0)
	assert.InDelta(t, 500, median, 20)

	multi, err := compute.TDigest(ctx, compute.TDigestOptions{Q: []float64{0.25, 0.5, 0.75}, SkipNulls: true},
		&compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer multi.Release()
	multiArr := multi.(*compute.ArrayDatum).MakeArray()
	defer multiArr.Release()
	assert.Equal(t, 3, multiArr.Len())
	assert.InDelta(t, 250, multiArr.(*array.Float64).Value(0), 25)
	assert.InDelta(t, 750, multiArr.(*array.Float64).Value(2), 25)
}

func TestTDigestEmpty(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)
	ctx := compute.WithAllocator(context.Background(), mem)

	in := aggArray(t, mem, arrow.PrimitiveTypes.Int32, `[]`)
	defer in.Release()
	res, err := compute.TDigest(ctx, compute.DefaultTDigestOptions(), &compute.ArrayDatum{Value: in.Data()})
	require.NoError(t, err)
	defer res.Release()
	arr := res.(*compute.ArrayDatum).MakeArray()
	defer arr.Release()
	assert.True(t, arr.IsNull(0))
}
