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

// Package tdigest implements approximate quantiles over an arbitrary length
// stream in O(1) space using the t-digest algorithm described in
// "Computing Extremely Accurate Quantiles Using t-Digests" by Dunning & Ertl.
package tdigest

import (
	"math"
	"sort"
)

const twoPi = 2 * math.Pi

func lerp(a, b, t float64) float64 { return a + t*(b-a) }

type centroid struct {
	mean   float64
	weight float64
}

func (c *centroid) merge(other centroid) {
	c.weight += other.weight
	c.mean += (other.mean - c.mean) * other.weight / c.weight
}

// scalerK1 is the K1 scale function.
type scalerK1 struct{ deltaNorm float64 }

func (s scalerK1) k(q float64) float64 { return s.deltaNorm * math.Asin(2*q-1) }
func (s scalerK1) q(k float64) float64 { return (math.Sin(k/s.deltaNorm) + 1) / 2 }

// merger implements the t-digest merging algorithm.
type merger struct {
	scalerK1

	totalWeight float64
	weightSoFar float64
	weightLimit float64
	tdigest     *[]centroid
}

func newMerger(delta uint32) merger {
	m := merger{scalerK1: scalerK1{deltaNorm: float64(delta) / twoPi}}
	m.reset(0, nil)
	return m
}

func (m *merger) reset(totalWeight float64, td *[]centroid) {
	m.totalWeight = totalWeight
	m.tdigest = td
	if td != nil {
		*td = (*td)[:0]
	}
	m.weightSoFar = 0
	m.weightLimit = -1
}

func (m *merger) add(c centroid) {
	td := *m.tdigest
	weight := m.weightSoFar + c.weight
	if weight <= m.weightLimit {
		td[len(td)-1].merge(c)
	} else {
		quantile := m.weightSoFar / m.totalWeight
		nextWeightLimit := m.totalWeight * m.q(m.k(quantile)+1)
		if nextWeightLimit <= m.weightLimit {
			m.weightLimit = m.totalWeight
		} else {
			m.weightLimit = nextWeightLimit
		}
		td = append(td, c)
	}
	m.weightSoFar = weight
	*m.tdigest = td
}

type tdigestImpl struct {
	delta       uint32
	merger      merger
	totalWeight float64
	min, max    float64
	tdigests    [2][]centroid
	current     int
}

func newImpl(delta uint32) *tdigestImpl {
	if delta <= 10 {
		delta = 10
	}
	impl := &tdigestImpl{
		delta:    delta,
		merger:   newMerger(delta),
		tdigests: [2][]centroid{make([]centroid, 0, delta), make([]centroid, 0, delta)},
	}
	impl.reset()
	return impl
}

func (t *tdigestImpl) reset() {
	t.tdigests[0] = t.tdigests[0][:0]
	t.tdigests[1] = t.tdigests[1][:0]
	t.current = 0
	t.totalWeight = 0
	t.min = math.MaxFloat64
	t.max = -math.MaxFloat64
	t.merger.reset(0, nil)
}

func (t *tdigestImpl) mergeInput(input []float64) {
	t.totalWeight += float64(len(input))
	sort.Float64s(input)
	if input[0] < t.min {
		t.min = input[0]
	}
	if input[len(input)-1] > t.max {
		t.max = input[len(input)-1]
	}

	dst := 1 - t.current
	t.merger.reset(t.totalWeight, &t.tdigests[dst])
	td := t.tdigests[t.current]
	i, j := 0, 0
	for i < len(td) && j < len(input) {
		if td[i].mean < input[j] {
			t.merger.add(td[i])
			i++
		} else {
			t.merger.add(centroid{mean: input[j], weight: 1})
			j++
		}
	}
	for i < len(td) {
		t.merger.add(td[i])
		i++
	}
	for j < len(input) {
		t.merger.add(centroid{mean: input[j], weight: 1})
		j++
	}
	t.merger.reset(0, nil)
	t.current = dst
}

func (t *tdigestImpl) merge(others []*tdigestImpl) {
	type source struct {
		td  []centroid
		idx int
	}
	srcs := make([]source, 0, len(others)+1)
	if len(t.tdigests[t.current]) > 0 {
		srcs = append(srcs, source{t.tdigests[t.current], 0})
	}
	for _, o := range others {
		otd := o.tdigests[o.current]
		if len(otd) == 0 {
			continue
		}
		srcs = append(srcs, source{otd, 0})
		t.totalWeight += o.totalWeight
		if o.min < t.min {
			t.min = o.min
		}
		if o.max > t.max {
			t.max = o.max
		}
	}

	dst := 1 - t.current
	t.merger.reset(t.totalWeight, &t.tdigests[dst])
	for {
		minIdx := -1
		for i := range srcs {
			if srcs[i].idx >= len(srcs[i].td) {
				continue
			}
			if minIdx == -1 || srcs[i].td[srcs[i].idx].mean < srcs[minIdx].td[srcs[minIdx].idx].mean {
				minIdx = i
			}
		}
		if minIdx == -1 {
			break
		}
		t.merger.add(srcs[minIdx].td[srcs[minIdx].idx])
		srcs[minIdx].idx++
	}
	t.merger.reset(0, nil)
	t.current = dst
}

func (t *tdigestImpl) quantile(q float64) float64 {
	td := t.tdigests[t.current]
	if q < 0 || q > 1 || len(td) == 0 {
		return math.NaN()
	}

	index := q * t.totalWeight
	if index <= 1 {
		return t.min
	} else if index >= t.totalWeight-1 {
		return t.max
	}

	ci := 0
	weightSum := 0.0
	for ; ci < len(td); ci++ {
		weightSum += td[ci].weight
		if index <= weightSum {
			break
		}
	}

	diff := index + td[ci].weight/2 - weightSum
	if td[ci].weight == 1 && math.Abs(diff) < 0.5 {
		return td[ci].mean
	}

	ciLeft, ciRight := ci, ci
	if diff > 0 {
		if ciRight == len(td)-1 {
			c := td[ciRight]
			return lerp(c.mean, t.max, diff/(c.weight/2))
		}
		ciRight++
	} else {
		if ciLeft == 0 {
			c := td[0]
			return lerp(t.min, c.mean, index/(c.weight/2))
		}
		ciLeft--
		diff += td[ciLeft].weight/2 + td[ciRight].weight/2
	}

	diff /= td[ciLeft].weight/2 + td[ciRight].weight/2
	return lerp(td[ciLeft].mean, td[ciRight].mean, diff)
}

func (t *tdigestImpl) mean() float64 {
	sum := 0.0
	for _, c := range t.tdigests[t.current] {
		sum += c.mean * c.weight
	}
	if t.totalWeight == 0 {
		return math.NaN()
	}
	return sum / t.totalWeight
}

// TDigest is an approximate quantile sketch.
type TDigest struct {
	input      []float64
	bufferSize uint32
	impl       *tdigestImpl
}

// New returns a new TDigest with the given compression parameter (delta) and
// input buffer size. delta values below 10 are clamped to 10 and a buffer size
// of 0 defaults to 500.
func New(delta, bufferSize uint32) *TDigest {
	if bufferSize == 0 {
		bufferSize = 500
	}
	return &TDigest{
		input:      make([]float64, 0, bufferSize),
		bufferSize: bufferSize,
		impl:       newImpl(delta),
	}
}

// Add buffers a single data point, ignoring NaN values.
func (t *TDigest) Add(value float64) {
	if math.IsNaN(value) {
		return
	}
	if uint32(len(t.input)) == t.bufferSize {
		t.mergeInput()
	}
	t.input = append(t.input, value)
}

func (t *TDigest) mergeInput() {
	if len(t.input) > 0 {
		t.impl.mergeInput(t.input)
		t.input = t.input[:0]
	}
}

// Merge merges another TDigest into this one.
func (t *TDigest) Merge(other *TDigest) {
	t.mergeInput()
	other.mergeInput()
	t.impl.merge([]*tdigestImpl{other.impl})
}

// Quantile returns the approximate quantile for q in [0, 1].
func (t *TDigest) Quantile(q float64) float64 {
	t.mergeInput()
	return t.impl.quantile(q)
}

// Mean returns the approximate mean.
func (t *TDigest) Mean() float64 {
	t.mergeInput()
	return t.impl.mean()
}

// IsEmpty reports whether no data points have been added.
func (t *TDigest) IsEmpty() bool {
	return len(t.input) == 0 && t.impl.totalWeight == 0
}
