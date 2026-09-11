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

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// Aggregate describes a single aggregation to compute in GroupBy.
type Aggregate struct {
	// Function is the name of the scalar aggregate function, for example
	// "sum", "mean", "min", "max", "count" or "count_distinct".
	Function string
	// Options are the function options (for example ScalarAggregateOptions or
	// CountOptions). May be nil to use the function defaults.
	Options FunctionOptions
	// Value is the array (or chunked array) of values to aggregate.
	Value Datum
}

// buildKeyColumn builds one output key column with one value per group by
// taking the first row of each group.
func buildKeyColumn(mem memory.Allocator, key arrow.Array, groups [][]int64) (arrow.Array, error) {
	bldr := array.NewBuilder(mem, key.DataType())
	defer bldr.Release()
	for _, rows := range groups {
		sc, err := scalar.GetScalar(key, int(rows[0]))
		if err != nil {
			return nil, err
		}
		if err := scalar.Append(bldr, sc); err != nil {
			return nil, err
		}
	}
	return bldr.NewArray(), nil
}

// GroupBy computes grouped aggregations over one or more key columns and
// returns a table with one row per group. The output columns are the key
// columns (named key_0, key_1, ...) followed by one column per aggregate,
// named after the aggregate function. Groups are ordered by first appearance.
func GroupBy(ctx context.Context, keys []Datum, aggregates []Aggregate) (arrow.Table, error) {
	keyArrs := make([]arrow.Array, len(keys))
	for i, k := range keys {
		arr, err := datumToArray(ctx, k)
		if err != nil {
			return nil, err
		}
		defer arr.Release()
		keyArrs[i] = arr
	}

	var length int
	if len(keyArrs) > 0 {
		length = keyArrs[0].Len()
	}
	for _, k := range keyArrs {
		if k.Len() != length {
			return nil, fmt.Errorf("%w: group key length %d does not match %d", arrow.ErrInvalid, k.Len(), length)
		}
	}

	groups, err := groupRows(keyArrs, length)
	if err != nil {
		return nil, err
	}
	numGroups := len(groups)
	mem := exec.GetAllocator(ctx)

	fields := make([]arrow.Field, 0, len(keyArrs)+len(aggregates))
	cols := make([]arrow.Array, 0, len(keyArrs)+len(aggregates))
	defer func() {
		for _, c := range cols {
			c.Release()
		}
	}()

	for i, k := range keyArrs {
		col, err := buildKeyColumn(mem, k, groups)
		if err != nil {
			return nil, err
		}
		cols = append(cols, col)
		fields = append(fields, arrow.Field{
			Name: fmt.Sprintf("key_%d", i), Type: k.DataType(), Nullable: true})
	}

	for _, agg := range aggregates {
		valueArr, err := datumToArray(ctx, agg.Value)
		if err != nil {
			return nil, err
		}
		if valueArr.Len() != length {
			valueArr.Release()
			return nil, fmt.Errorf("%w: aggregate value length %d does not match key length %d",
				arrow.ErrInvalid, valueArr.Len(), length)
		}
		out, err := hashAggregateScalar(ctx, agg.Function, agg.Options, valueArr, keyArrs)
		valueArr.Release()
		if err != nil {
			return nil, err
		}
		cols = append(cols, out)
		fields = append(fields, arrow.Field{Name: agg.Function, Type: out.DataType(), Nullable: true})
	}

	schema := arrow.NewSchema(fields, nil)
	batch := array.NewRecordBatch(schema, cols, int64(numGroups))
	defer batch.Release()
	return array.NewTableFromRecords(schema, []arrow.RecordBatch{batch}), nil
}
