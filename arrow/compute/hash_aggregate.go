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
	"encoding/binary"
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// encodeGroupKey appends a stable encoding of the given scalar to buf. Values
// are length-prefixed so that adjacent key columns cannot be confused, and a
// leading validity marker distinguishes nulls.
func encodeGroupKey(buf []byte, sc scalar.Scalar) []byte {
	if !sc.IsValid() {
		return append(buf, 0)
	}
	buf = append(buf, 1)

	switch s := sc.(type) {
	case *scalar.Boolean:
		if s.Value {
			return append(buf, 1)
		}
		return append(buf, 0)
	case scalar.PrimitiveScalar:
		data := s.Data()
		buf = binary.AppendUvarint(buf, uint64(len(data)))
		return append(buf, data...)
	case scalar.BinaryScalar:
		data := s.Data()
		buf = binary.AppendUvarint(buf, uint64(len(data)))
		return append(buf, data...)
	default:
		str := sc.String()
		buf = binary.AppendUvarint(buf, uint64(len(str)))
		return append(buf, str...)
	}
}

// groupRows maps each row of the key arrays to a contiguous group id in the
// order groups are first encountered. It returns the list of row indices for
// each group.
func groupRows(keys []arrow.Array, length int) ([][]int64, error) {
	if length == 0 {
		return nil, nil
	}
	ids := make(map[string]int32)
	var groups [][]int64
	buf := make([]byte, 0, 32)
	for i := 0; i < length; i++ {
		buf = buf[:0]
		for _, k := range keys {
			sc, err := scalar.GetScalar(k, i)
			if err != nil {
				return nil, err
			}
			buf = encodeGroupKey(buf, sc)
		}
		id, ok := ids[string(buf)]
		if !ok {
			id = int32(len(groups))
			ids[string(buf)] = id
			groups = append(groups, nil)
		}
		groups[id] = append(groups[id], int64(i))
	}
	return groups, nil
}

func datumToArray(ctx context.Context, d Datum) (arrow.Array, error) {
	switch v := d.(type) {
	case *ArrayDatum:
		return v.MakeArray(), nil
	case *ChunkedDatum:
		return array.Concatenate(v.Value.Chunks(), exec.GetAllocator(ctx))
	case *ScalarDatum:
		return scalar.MakeArrayFromScalar(v.Value, 1, exec.GetAllocator(ctx))
	default:
		return nil, fmt.Errorf("%w: unsupported datum for hash aggregate: %s", arrow.ErrInvalid, d)
	}
}

func takeGroup(ctx context.Context, values arrow.Array, rows []int64) (arrow.Array, error) {
	mem := exec.GetAllocator(ctx)
	bldr := array.NewInt64Builder(mem)
	defer bldr.Release()
	bldr.AppendValues(rows, nil)
	indices := bldr.NewArray()
	defer indices.Release()
	return TakeArray(ctx, values, indices)
}

// hashAggregateScalar runs the named scalar aggregate over each group of
// values and assembles the per-group results into a single array.
func hashAggregateScalar(ctx context.Context, aggName string, opts FunctionOptions, values arrow.Array, keys []arrow.Array) (arrow.Array, error) {
	groups, err := groupRows(keys, values.Len())
	if err != nil {
		return nil, err
	}
	mem := exec.GetAllocator(ctx)

	if len(groups) == 0 {
		sc, err := CallFunction(ctx, aggName, opts, &ArrayDatum{Value: values.Data()})
		if err != nil {
			return nil, err
		}
		defer sc.Release()
		outType := sc.(ArrayLikeDatum).Type()
		return array.MakeArrayOfNull(mem, outType, 0), nil
	}

	var bldr array.Builder
	defer func() {
		if bldr != nil {
			bldr.Release()
		}
	}()

	for _, rows := range groups {
		taken, err := takeGroup(ctx, values, rows)
		if err != nil {
			return nil, err
		}
		sc, err := CallFunction(ctx, aggName, opts, &ArrayDatum{Value: taken.Data()})
		taken.Release()
		if err != nil {
			return nil, err
		}
		scalarVal := sc.(*ScalarDatum).Value
		if bldr == nil {
			bldr = array.NewBuilder(mem, scalarVal.DataType())
		}
		if err := scalar.Append(bldr, scalarVal); err != nil {
			sc.Release()
			return nil, err
		}
		sc.Release()
	}
	result := bldr.NewArray()
	bldr.Release()
	bldr = nil
	return result, nil
}

// hashAggregateList builds a list array where each element contains the values
// of one group. If distinct is true, duplicate values within a group are
// removed.
func hashAggregateList(ctx context.Context, values arrow.Array, keys []arrow.Array, distinct bool) (arrow.Array, error) {
	groups, err := groupRows(keys, values.Len())
	if err != nil {
		return nil, err
	}
	mem := exec.GetAllocator(ctx)
	lb := array.NewListBuilder(mem, values.DataType())
	defer lb.Release()

	for _, rows := range groups {
		taken, err := takeGroup(ctx, values, rows)
		if err != nil {
			return nil, err
		}
		vb := lb.ValueBuilder()
		seen := make(map[string]struct{})
		for i := 0; i < taken.Len(); i++ {
			sc, err := scalar.GetScalar(taken, i)
			if err != nil {
				taken.Release()
				return nil, err
			}
			if distinct {
				buf := encodeGroupKey(nil, sc)
				if _, ok := seen[string(buf)]; ok {
					continue
				}
				seen[string(buf)] = struct{}{}
			}
			if err := scalar.Append(vb, sc); err != nil {
				taken.Release()
				return nil, err
			}
		}
		taken.Release()
		lb.Append(true)
	}
	return lb.NewArray(), nil
}

func hashAggregate(ctx context.Context, aggName string, opts FunctionOptions, values Datum, keys ...Datum) (Datum, error) {
	valueArr, err := datumToArray(ctx, values)
	if err != nil {
		return nil, err
	}
	defer valueArr.Release()

	keyArrs := make([]arrow.Array, len(keys))
	for i, k := range keys {
		keyArrs[i], err = datumToArray(ctx, k)
		if err != nil {
			return nil, err
		}
		defer keyArrs[i].Release()
		if keyArrs[i].Len() != valueArr.Len() {
			return nil, fmt.Errorf("%w: hash aggregate key length %d does not match value length %d",
				arrow.ErrInvalid, keyArrs[i].Len(), valueArr.Len())
		}
	}

	if len(keyArrs) == 0 {
		return CallFunction(ctx, aggName, opts, values)
	}

	out, err := hashAggregateScalar(ctx, aggName, opts, valueArr, keyArrs)
	if err != nil {
		return nil, err
	}
	defer out.Release()
	return NewDatum(out), nil
}

// HashCount counts the number of values in each group.
func HashCount(ctx context.Context, opts CountOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "count", &opts, values, keys...)
}

// HashCountDistinct counts the number of distinct values in each group.
func HashCountDistinct(ctx context.Context, opts CountOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "count_distinct", &opts, values, keys...)
}

// HashSum computes the sum of values in each group.
func HashSum(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "sum", &opts, values, keys...)
}

// HashMean computes the mean of values in each group.
func HashMean(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "mean", &opts, values, keys...)
}

// HashMin computes the minimum value in each group.
func HashMin(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "min", &opts, values, keys...)
}

// HashMax computes the maximum value in each group.
func HashMax(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "max", &opts, values, keys...)
}

// HashMinMax computes the minimum and maximum values in each group.
func HashMinMax(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "min_max", &opts, values, keys...)
}

// HashFirst computes the first value in each group.
func HashFirst(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "first", &opts, values, keys...)
}

// HashLast computes the last value in each group.
func HashLast(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "last", &opts, values, keys...)
}

// HashOne returns one value from each group.
func HashOne(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "first", &opts, values, keys...)
}

// HashAny tests whether any value in each group is true.
func HashAny(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "any", &opts, values, keys...)
}

// HashAll tests whether all values in each group are true.
func HashAll(ctx context.Context, opts ScalarAggregateOptions, values Datum, keys ...Datum) (Datum, error) {
	return hashAggregate(ctx, "all", &opts, values, keys...)
}

// HashDistinct returns the distinct values of each group as a list array.
func HashDistinct(ctx context.Context, values Datum, keys ...Datum) (Datum, error) {
	valueArr, err := datumToArray(ctx, values)
	if err != nil {
		return nil, err
	}
	defer valueArr.Release()
	keyArrs := make([]arrow.Array, len(keys))
	for i, k := range keys {
		keyArrs[i], err = datumToArray(ctx, k)
		if err != nil {
			return nil, err
		}
		defer keyArrs[i].Release()
	}
	out, err := hashAggregateList(ctx, valueArr, keyArrs, true)
	if err != nil {
		return nil, err
	}
	defer out.Release()
	return NewDatum(out), nil
}

// HashList returns all values of each group as a list array.
func HashList(ctx context.Context, values Datum, keys ...Datum) (Datum, error) {
	valueArr, err := datumToArray(ctx, values)
	if err != nil {
		return nil, err
	}
	defer valueArr.Release()
	keyArrs := make([]arrow.Array, len(keys))
	for i, k := range keys {
		keyArrs[i], err = datumToArray(ctx, k)
		if err != nil {
			return nil, err
		}
		defer keyArrs[i].Release()
	}
	out, err := hashAggregateList(ctx, valueArr, keyArrs, false)
	if err != nil {
		return nil, err
	}
	defer out.Release()
	return NewDatum(out), nil
}
