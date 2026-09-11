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

	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/compute/internal/kernels"
)

// CountMode is re-exported from the internal kernels package.
type CountMode = kernels.CountMode

// CountOptions controls count aggregate kernel behavior.
type CountOptions = kernels.CountOptions

// ScalarAggregateOptions controls the behavior of the scalar aggregate
// functions.
type ScalarAggregateOptions = kernels.ScalarAggregateOptions

// VarianceOptions controls the variance and stddev kernels.
type VarianceOptions = kernels.VarianceOptions

// IndexOptions controls the index aggregate kernel.
type IndexOptions = kernels.IndexOptions

const (
	// CountOnlyValid counts only non-null values. This is the default.
	CountOnlyValid = kernels.CountOnlyValid
	// CountOnlyNull counts only null values.
	CountOnlyNull = kernels.CountOnlyNull
	// CountAll counts both non-null and null values.
	CountAll = kernels.CountAll
)

// DefaultScalarAggregateOptions returns the default scalar aggregate options
// (skip nulls, min_count zero).
func DefaultScalarAggregateOptions() ScalarAggregateOptions {
	return kernels.DefaultScalarAggregateOptions()
}

// DefaultVarianceOptions returns the default variance options (skip nulls,
// ddof zero, min_count zero).
func DefaultVarianceOptions() VarianceOptions {
	return kernels.DefaultVarianceOptions()
}

var (
	countDoc = FunctionDoc{
		Summary:     "Count the number of null / non-null values",
		Description: "By default, only non-null values are counted.\nThis can be changed through CountOptions.",
		ArgNames:    []string{"array"},
		OptionsType: "CountOptions",
	}
	countDistinctDoc = FunctionDoc{
		Summary:     "Count the number of unique values",
		Description: "By default, only non-null values are counted.\nThis can be changed through CountOptions.",
		ArgNames:    []string{"array"},
		OptionsType: "CountOptions",
	}
	sumDoc = FunctionDoc{
		Summary: "Compute the sum of a numeric array",
		Description: `Null values are ignored by default. Minimum count of non-null
values can be set and null is returned if too few are present.
This can be changed through ScalarAggregateOptions.`,
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	meanDoc = FunctionDoc{
		Summary: "Compute the mean of a numeric array",
		Description: `Null values are ignored by default. Minimum count of non-null
values can be set and null is returned if too few are present.
This can be changed through ScalarAggregateOptions.
For integers and floats, NaN is returned if min_count = 0 and there are
no values.`,
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	productDoc = FunctionDoc{
		Summary: "Compute the product of values in a numeric array",
		Description: `Null values are ignored by default. Minimum count of non-null
values can be set and null is returned if too few are present.
This can be changed through ScalarAggregateOptions.`,
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	minMaxDoc = FunctionDoc{
		Summary:     "Compute the minimum and maximum values of a numeric array",
		Description: "Null values are ignored by default.\nThis can be changed through ScalarAggregateOptions.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	minOrMaxDoc = FunctionDoc{
		Summary:     "Compute the minimum or maximum values of a numeric array",
		Description: "Null values are ignored by default.\nThis can be changed through ScalarAggregateOptions.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	anyDoc = FunctionDoc{
		Summary:     "Test whether any element in a boolean array evaluates to true",
		Description: "Null values are ignored by default.\nIf skip_nulls is false, then Kleene logic is used.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	allDoc = FunctionDoc{
		Summary:     "Test whether all elements in a boolean array evaluate to true",
		Description: "Null values are ignored by default.\nIf skip_nulls is false, then Kleene logic is used.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	firstLastDoc = FunctionDoc{
		Summary:     "Compute the first and last values of an array",
		Description: "Null values are ignored by default.\nIf skip_nulls is false, the first and last values are returned even if null.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	firstDoc = FunctionDoc{
		Summary:     "Compute the first value of an array",
		Description: "Null values are ignored by default.\nIf skip_nulls is false, the first value is returned even if null.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	lastDoc = FunctionDoc{
		Summary:     "Compute the last value of an array",
		Description: "Null values are ignored by default.\nIf skip_nulls is false, the last value is returned even if null.",
		ArgNames:    []string{"array"},
		OptionsType: "ScalarAggregateOptions",
	}
	indexDoc = FunctionDoc{
		Summary:         "Find the index of the first occurrence of a given value",
		Description:     "-1 is returned if the value is not found in the array.\nThe search value is specified in IndexOptions.",
		ArgNames:        []string{"array"},
		OptionsType:     "IndexOptions",
		OptionsRequired: true,
	}
	varianceDoc = FunctionDoc{
		Summary:     "Compute the variance of a numeric array",
		Description: "Null values are ignored by default.\nThis can be changed through VarianceOptions.",
		ArgNames:    []string{"array"},
		OptionsType: "VarianceOptions",
	}
	stddevDoc = FunctionDoc{
		Summary:     "Compute the standard deviation of a numeric array",
		Description: "Null values are ignored by default.\nThis can be changed through VarianceOptions.",
		ArgNames:    []string{"array"},
		OptionsType: "VarianceOptions",
	}
)

func registerAggFunction(reg FunctionRegistry, name string, doc FunctionDoc, kernelsList []exec.ScalarAggregateKernel, defaultOpts FunctionOptions) {
	fn := NewScalarAggregateFunction(name, Unary(), doc)
	if defaultOpts != nil {
		fn.SetDefaultOptions(defaultOpts)
	}
	for _, k := range kernelsList {
		if err := fn.AddKernel(k); err != nil {
			panic(err)
		}
	}
	if !reg.AddFunction(fn, false) {
		panic(fmt.Errorf("function '%s' already exists", name))
	}
}

// RegisterScalarAggregate registers the scalar aggregate functions with the
// provided registry.
func RegisterScalarAggregate(reg FunctionRegistry) {
	k := kernels.GetScalarAggregateKernels()
	defaultAgg := DefaultScalarAggregateOptions()
	defaultVar := DefaultVarianceOptions()
	defaultCount := &CountOptions{}

	registerAggFunction(reg, "count", countDoc, k.Count, defaultCount)
	registerAggFunction(reg, "count_distinct", countDistinctDoc, k.CountDistinct, defaultCount)
	registerAggFunction(reg, "sum", sumDoc, k.Sum, &defaultAgg)
	registerAggFunction(reg, "mean", meanDoc, k.Mean, &defaultAgg)
	registerAggFunction(reg, "product", productDoc, k.Product, &defaultAgg)
	registerAggFunction(reg, "min_max", minMaxDoc, k.MinMax, &defaultAgg)
	registerAggFunction(reg, "min", minOrMaxDoc, k.Min, &defaultAgg)
	registerAggFunction(reg, "max", minOrMaxDoc, k.Max, &defaultAgg)
	registerAggFunction(reg, "any", anyDoc, k.Any, &defaultAgg)
	registerAggFunction(reg, "all", allDoc, k.All, &defaultAgg)
	registerAggFunction(reg, "first_last", firstLastDoc, k.FirstLast, &defaultAgg)
	registerAggFunction(reg, "first", firstDoc, k.First, &defaultAgg)
	registerAggFunction(reg, "last", lastDoc, k.Last, &defaultAgg)
	registerAggFunction(reg, "index", indexDoc, k.Index, nil)
	registerAggFunction(reg, "variance", varianceDoc, k.Variance, &defaultVar)
	registerAggFunction(reg, "stddev", stddevDoc, k.StdDev, &defaultVar)
}

// Count returns the number of null / non-null values in values.
func Count(ctx context.Context, opts CountOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "count", &opts, values)
}

// CountDistinct returns the number of unique values in values.
func CountDistinct(ctx context.Context, opts CountOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "count_distinct", &opts, values)
}

// Sum computes the sum of a numeric array.
func Sum(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "sum", &opts, values)
}

// Mean computes the mean of a numeric array.
func Mean(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "mean", &opts, values)
}

// Product computes the product of values in a numeric array.
func Product(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "product", &opts, values)
}

// MinMax computes the minimum and maximum values of an array.
func MinMax(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "min_max", &opts, values)
}

// Min computes the minimum value of an array.
func Min(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "min", &opts, values)
}

// Max computes the maximum value of an array.
func Max(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "max", &opts, values)
}

// Any tests whether any element in a boolean array evaluates to true.
func Any(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "any", &opts, values)
}

// All tests whether all elements in a boolean array evaluate to true.
func All(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "all", &opts, values)
}

// FirstLast computes the first and last values of an array.
func FirstLast(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "first_last", &opts, values)
}

// First computes the first value of an array.
func First(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "first", &opts, values)
}

// Last computes the last value of an array.
func Last(ctx context.Context, opts ScalarAggregateOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "last", &opts, values)
}

// Index finds the index of the first occurrence of a given value.
func Index(ctx context.Context, opts IndexOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "index", &opts, values)
}

// Variance computes the variance of a numeric array.
func Variance(ctx context.Context, opts VarianceOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "variance", &opts, values)
}

// StdDev computes the standard deviation of a numeric array.
func StdDev(ctx context.Context, opts VarianceOptions, values Datum) (Datum, error) {
	return CallFunction(ctx, "stddev", &opts, values)
}
