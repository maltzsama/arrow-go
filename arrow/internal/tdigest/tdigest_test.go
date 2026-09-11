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

package tdigest

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQuantiles(t *testing.T) {
	td := New(100, 500)
	for i := 1; i <= 10000; i++ {
		td.Add(float64(i))
	}
	assert.False(t, td.IsEmpty())
	assert.InDelta(t, 5000, td.Quantile(0.5), 100)
	assert.InDelta(t, 2500, td.Quantile(0.25), 100)
	assert.InDelta(t, 7500, td.Quantile(0.75), 100)
	assert.InDelta(t, 1, td.Quantile(0), 1)
	assert.InDelta(t, 10000, td.Quantile(1), 1)
}

func TestMerge(t *testing.T) {
	a := New(100, 500)
	b := New(100, 500)
	for i := 1; i <= 5000; i++ {
		a.Add(float64(i))
	}
	for i := 5001; i <= 10000; i++ {
		b.Add(float64(i))
	}
	a.Merge(b)
	assert.InDelta(t, 5000, a.Quantile(0.5), 100)
}

func TestEmpty(t *testing.T) {
	td := New(100, 500)
	assert.True(t, td.IsEmpty())
	// NaN values are ignored.
	td.Add(1)
	td.Add(2)
	td.Add(3)
	assert.False(t, td.IsEmpty())
	assert.InDelta(t, 2, td.Quantile(0.5), 1)
}
