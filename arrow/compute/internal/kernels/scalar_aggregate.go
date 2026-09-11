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

package kernels

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/scalar"
)

// CountMode controls the behavior of the count aggregate kernel.
type CountMode int8

const (
	// CountOnlyValid counts only non-null values. This is the default.
	CountOnlyValid CountMode = iota
	// CountOnlyNull counts only null values.
	CountOnlyNull
	// CountAll counts both non-null and null values.
	CountAll
)

// CountOptions controls count aggregate kernel behavior.
//
// By default, only non-null values are counted.
type CountOptions struct {
	Mode CountMode `compute:"mode"`
}

func (CountOptions) TypeName() string { return "CountOptions" }

type countState struct {
	mode  CountMode
	count int64
}

func initCount(_ *exec.KernelCtx, args exec.KernelInitArgs) (exec.KernelState, error) {
	opts := CountOptions{}
	switch value := args.Options.(type) {
	case nil:
	case CountOptions:
		opts = value
	case *CountOptions:
		if value != nil {
			opts = *value
		}
	default:
		return nil, fmt.Errorf("%w: attempted to initialize count from invalid function options", arrow.ErrInvalid)
	}

	return &countState{mode: opts.Mode}, nil
}

// countNulls returns the number of null values in the given ExecValue, which is
// expected to have a length of `length` elements.
func countNulls(val *exec.ExecValue, length int64) int64 {
	if val.IsScalar() {
		if val.Scalar.IsValid() {
			return 0
		}
		return length
	}

	arr := &val.Array
	if arr.Type.ID() == arrow.NULL {
		// Null arrays have no validity bitmap but every value is null.
		return length
	}
	if arr.Nulls == 0 || arr.Buffers[0].Buf == nil {
		return 0
	}
	if arr.Nulls == arr.Len {
		return length
	}
	return arr.UpdateNullCount()
}

func consumeCount(ctx *exec.KernelCtx, span *exec.ExecSpan) error {
	state, ok := ctx.State.(*countState)
	if !ok {
		return fmt.Errorf("%w: invalid kernel state for count", arrow.ErrInvalid)
	}

	if span.Len == 0 || len(span.Values) == 0 {
		return nil
	}

	nulls := countNulls(&span.Values[0], span.Len)
	switch state.mode {
	case CountAll:
		state.count += span.Len
	case CountOnlyNull:
		state.count += nulls
	default: // CountOnlyValid
		state.count += span.Len - nulls
	}
	return nil
}

func mergeCount(_ *exec.KernelCtx, src exec.KernelState, dst *exec.KernelState) error {
	other, ok := src.(*countState)
	if !ok {
		return fmt.Errorf("%w: invalid source kernel state for count", arrow.ErrInvalid)
	}
	state, ok := (*dst).(*countState)
	if !ok {
		return fmt.Errorf("%w: invalid destination kernel state for count", arrow.ErrInvalid)
	}

	state.count += other.count
	return nil
}

func finalizeCount(ctx *exec.KernelCtx) (scalar.Scalar, error) {
	state, ok := ctx.State.(*countState)
	if !ok {
		return nil, fmt.Errorf("%w: invalid kernel state for count", arrow.ErrInvalid)
	}
	return scalar.NewInt64Scalar(state.count), nil
}

// GetScalarAggregateKernels returns the registered set of scalar aggregate
// kernels. Each returned slice corresponds to a single function's kernels.
func GetScalarAggregateKernels() (count []exec.ScalarAggregateKernel) {
	count = append(count, exec.NewScalarAggregateKernel(
		[]exec.InputType{{Kind: exec.InputAny}},
		exec.NewOutputType(arrow.PrimitiveTypes.Int64),
		initCount, consumeCount, mergeCount, finalizeCount, false,
	))
	return
}
