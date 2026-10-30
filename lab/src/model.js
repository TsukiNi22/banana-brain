// Simulations of the networks of banana-brain, same rules as the C++ code (LayerNetwork forward_ / learn_, RawBrainNetwork init_ / apply / learn_ / relink_)
// The options of the lab (activation, link modes, protections...) default to the behavior of the library
function rng(seed) { var a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function sigm(x) { return 1 / (1 + Math.exp(-x)); }
var ACT = { // activation and its derivative at the sum (output = activation of the sum), as ANetwork::activate_ / derivative_
  sigmoid: { f: sigm, d: function (s, o) { return o * (1 - o); } },
  relu: { f: function (s) { return s > 0 ? s : 0; }, d: function (s) { return s > 0 ? 1 : 0; } },
  linear: { f: function (s) { return s; }, d: function () { return 1; } }
};

// ---------- layered network ----------
function LayerSim(sizes, seed, opt) { // sizes = [inputs, layer1, ..., output]
  opt = opt || {}; var r = rng(seed), range = opt.range || 1, rw = opt.weightInit !== false, rb = opt.biasInit !== false;
  this.act = ACT[opt.activation || 'sigmoid']; this.sizes = sizes; this.layers = [];
  for (var l = 1; l < sizes.length; l++) {
    var layer = [];
    for (var k = 0; k < sizes[l]; k++) { var w = []; for (var j = 0; j < sizes[l - 1]; j++) w.push(rw ? (r() * 2 - 1) * range : 1); layer.push({ w: w, b: rb ? (r() * 2 - 1) * range : 0 }); }
    this.layers.push(layer);
  }
}
LayerSim.prototype.forward = function (x) {
  if (x.length && Array.isArray(x[0])) x = [].concat.apply([], x); // a sequence: the layered network sees all its steps at once
  var outs = [x.slice()], sums = [], act = this.act;
  for (var l = 0; l < this.layers.length; l++) {
    var prev = outs[l], s = [], o = [];
    this.layers[l].forEach(function (n) { var v = n.b; for (var j = 0; j < prev.length; j++) v += n.w[j] * prev[j]; s.push(v); o.push(act.f(v)); });
    sums.push(s); outs.push(o);
  }
  this.last = { outs: outs, sums: sums }; return outs[outs.length - 1];
};
LayerSim.prototype.learn = function (errors) { // errors = rate * (output - expected), classic descent
  var outs = this.last.outs, sums = this.last.sums, delta = errors.slice(), changes = [], act = this.act;
  for (var l = this.layers.length - 1; l >= 0; l--) {
    var prev = outs[l], below = new Array(prev.length).fill(0), ch = [];
    for (var k = 0; k < this.layers[l].length; k++) {
      var n = this.layers[l][k], slope = delta[k] * act.d(sums[l][k], outs[l + 1][k]), row = [];
      for (var j = 0; j < prev.length; j++) { below[j] += slope * n.w[j]; var d = slope * prev[j]; n.w[j] -= d; row.push(-d); }
      n.b -= slope; ch.push({ w: row, b: -slope });
    }
    changes.unshift(ch); delta = below;
  }
  return changes;
};

// ---------- raw brain network ----------
// opt: seed, neurons, input, output, cap, steps, min, max, width, height, range, threshold, activation,
//      weightInit / biasInit (false: 1.0 / 0.0), linkInit ('max' | 'random': 1 -> cap), minLinks, direction ('random' | 'forward': from the left to the right),
//      relink ('normal' | 'inverted' | 'off'), protectPath (never cut the last path of an input to the output), orphans
function BrainSim(opt) {
  var r = this.r = rng(opt.seed); this.n = opt.neurons; this.input = opt.input; this.output = opt.output; this.opt = opt;
  this.cap = opt.cap; this.steps = opt.steps; this.min = opt.min; this.max = opt.max; this.threshold = opt.threshold || 0;
  this.act = ACT[opt.activation || 'sigmoid']; this.relinkMode = opt.relink || 'normal'; this.minLinks = Math.min(opt.minLinks || 0, opt.cap);
  var W = opt.width, H = opt.height, n = this.n, pos = [], range = opt.range || 1; // weightInit / biasInit in rand(-range, range)
  for (var i = 0; i < n; i++) {
    if (i < this.input) pos.push({ x: 0, y: (i + 0.5) * H / this.input });
    else if (i >= n - this.output) pos.push({ x: W, y: (i - (n - this.output) + 0.5) * H / this.output });
    else pos.push({ x: r() * W, y: r() * H });
  }
  this.pos = pos; this.bias = new Array(n).fill(0);
  if (opt.direction === 'forward') this.leftList = this.leftOf(); // left to right: the neurons on the left of each one, nearest first, computed once (RawBrainNetwork::leftOf_)
  this.links = []; for (var i2 = 0; i2 < n * this.cap; i2++) this.links.push(null); // {id, w, inward}
  var used = new Array(n).fill(0), far = Math.min(Math.max(0, Math.round(opt.longLinks || 0)), this.cap), nearCap = this.cap - far; // longLinks: slots kept for far neurons drawn at random
  for (var owner = 0; owner < n; owner++) {
    var target = opt.linkInit === 'random' ? Math.max(this.minLinks, Math.min(this.cap, 1 + Math.floor(r() * this.cap))) : this.cap; // as many links as the cap allows by default
    var order = this.near(owner);
    for (var q = 0; q < order.length && used[owner] < Math.min(target, nearCap); q++) {
      var other = order[q];
      if (this.slotOf(owner, other) >= 0 || used[other] >= nearCap) continue;
      var dir = this.direction(owner, other); if (dir === null) continue;
      this.links[owner * this.cap + used[owner]++] = { id: other, w: 1, inward: dir }; // 1.0 by default, the weightInit below draws them
      this.links[other * this.cap + used[other]++] = { id: owner, w: 1, inward: !dir };
    }
  }
  if (far > 0) for (var o2 = 0; o2 < n; o2++) { // the far links: random neurons of the allowed ones (the left ones in left to right)
    var pool = this.near(o2).slice(), tries2 = 0;
    while (used[o2] < this.cap && pool.length && tries2++ < 4 * n) {
      var pick = pool.splice(Math.floor(r() * pool.length), 1)[0];
      if (this.slotOf(o2, pick) >= 0 || used[pick] >= this.cap) continue;
      var dir2 = this.direction(o2, pick); if (dir2 === null) continue;
      this.links[o2 * this.cap + used[o2]++] = { id: pick, w: 1, inward: dir2 };
      this.links[pick * this.cap + used[pick]++] = { id: o2, w: 1, inward: !dir2 };
    }
  }
  if (opt.pathInit) for (var pass = 0; pass < 3 && this.hopsAll() < 0; pass++) this.ensurePaths(); // a freed slot can cut an earlier path: checked again
  this.steps = Math.max(this.steps, Math.ceil(this.farthest() * 1.1)); // the time steps are never under the farthest path + 10 % (the signal has to arrive)
  if (opt.weightInit !== false) for (var o = 0; o < n; o++) for (var s = 0; s < this.cap; s++) { var L = this.links[o * this.cap + s]; if (L && L.inward) { L.w = (r() * 2 - 1) * range || 0.01; this.clone(o, L).w = L.w; } }
  if (opt.biasInit !== false) for (var b = this.input; b < n; b++) this.bias[b] = (r() * 2 - 1) * range;
}
// Direction of a new link seen from the owner: true = the owner receives, false = it sends, null = impossible (two inputs, two outputs)
BrainSim.prototype.direction = function (owner, other) {
  var n = this.n, canReceive = owner >= this.input && other < n - this.output, canSend = owner < n - this.output && other >= this.input;
  if (!canReceive && !canSend) return null;
  if (this.opt.direction === 'forward') return canReceive && this.pos[other].x < this.pos[owner].x ? true : null; // left to right: the owner only receives from a neuron on its left
  return canReceive && (!canSend || this.r() < 0.5);
};
// Every input gets a path of at most pathHops inward links to every output: the missing ones take the chain of the smallest squared distances (a full neuron gives its farthest link up)
BrainSim.prototype.reach = function (src) { var n = this.n, cap = this.cap, d = new Array(n).fill(-1), q = [src], head = 0; d[src] = 0; while (head < q.length) { var u = q[head++]; for (var s = 0; s < cap; s++) { var L = this.links[u * cap + s]; if (L && !L.inward && d[L.id] < 0) { d[L.id] = d[u] + 1; q.push(L.id); } } } return d; }; // along the outward links (the clones): n x cap
BrainSim.prototype.pairs = function () { var c = 0; for (var i = 0; i < this.input; i++) { var d = this.reach(i); for (var o = this.n - this.output; o < this.n; o++) if (d[o] >= 0) c++; } return c; }; // pairs input -> output with a path
BrainSim.prototype.farthest = function () { var far = 0; for (var i = 0; i < this.input; i++) { var d = this.reach(i); for (var o = this.n - this.output; o < this.n; o++) if (d[o] > far) far = d[o]; } return far; }; // the farthest pair input -> output that has a path (the pairs without one do not count)
BrainSim.prototype.hopsAll = function () { var worst = 0; for (var i = 0; i < this.input; i++) { var d = this.reach(i); for (var o = this.n - this.output; o < this.n; o++) { if (d[o] < 0) return -1; worst = Math.max(worst, d[o]); } } return worst; }; // the farthest pair input -> output (-1: a pair without path)
BrainSim.prototype.ensurePaths = function () {
  var n = this.n, cap = this.cap, H = Math.max(2, Math.min(this.opt.pathHops || 3, this.steps)), self = this, kept = this.kept = this.kept || []; // kept: [sender, receiver] of the links of the paths already built, never given up
  for (var i = 0; i < this.input; i++) for (var o = n - this.output; o < n; o++) {
    var d = this.reach(i); if (d[o] >= 0 && d[o] <= this.steps) continue;
    var locked = new Array(n).fill(0); kept.forEach(function (p) { locked[p[0]]++; locked[p[1]]++; }); // a neuron full of kept links can't take a new one
    // cost[k][v]: smallest sum of squared distances from the input to v in k links (an existing inward link costs 0), the last hop must reach o
    var cost = [], from = [], best = -1;
    for (var k = 0; k <= H; k++) { cost.push(new Array(n).fill(Infinity)); from.push(new Array(n).fill(-1)); }
    cost[0][i] = 0;
    for (var k2 = 1; k2 <= H; k2++) for (var u = 0; u < n; u++) {
      if (!isFinite(cost[k2 - 1][u])) continue;
      for (var v = 0; v < n; v++) {
        if (v === u || v < this.input || (v >= n - this.output && v !== o) || (u >= n - this.output)) continue;
        var forward = this.opt.direction !== 'forward' || this.pos[u].x < this.pos[v].x;
        if (!forward) continue;
        var s = this.slotOf(v, u), L = s >= 0 ? this.links[v * cap + s] : null, c = L ? (L.inward ? 0 : Infinity) : (locked[u] >= cap || locked[v] >= cap ? Infinity : this.d2(u, v));
        if (cost[k2 - 1][u] + c < cost[k2][v]) { cost[k2][v] = cost[k2 - 1][u] + c; from[k2][v] = u; }
      }
    }
    for (var k3 = 2; k3 <= H; k3++) if (isFinite(cost[k3][o]) && (best < 0 || cost[k3][o] < cost[best][o])) best = k3; // at least one hidden neuron on the way
    if (best < 0) { if (isFinite(cost[1][o])) best = 1; else continue; }
    var path = [o]; for (var k4 = best, at = o; k4 > 0; k4--) { at = from[k4][at]; path.unshift(at); }
    for (var p = 0; p + 1 < path.length; p++) {
      var a = path[p], b = path[p + 1]; kept.push([a, b]); if (this.slotOf(b, a) >= 0) continue;
      [a, b].forEach(function (x) { if (self.free(x) < 0) self.dropFarthest(x, path); });
      if (this.free(a) < 0 || this.free(b) < 0) continue;
      var w = this.opt.weightInit !== false ? ((this.r() * 2 - 1) * (this.opt.range || 1) || 0.01) : 1;
      this.links[b * cap + this.free(b)] = { id: a, w: w, inward: true };
      this.links[a * cap + this.free(a)] = { id: b, w: w, inward: false };
    }
  }
};
BrainSim.prototype.dropFarthest = function (x, keep) { // frees a slot of x: its farthest link that is not on the path being built
  var best = -1, bd = -1;
  var kept = this.kept || [];
  for (var s = 0; s < this.cap; s++) { var L = this.links[x * this.cap + s]; if (!L || keep.indexOf(L.id) >= 0 || kept.some(function (p) { return (p[0] === x && p[1] === L.id) || (p[1] === x && p[0] === L.id); })) continue; var dd = this.d2(x, L.id); if (dd > bd) { bd = dd; best = s; } }
  if (best >= 0) this.removeLink(x, best);
};
BrainSim.prototype.d2 = function (a, b) { var dx = this.pos[a].x - this.pos[b].x, dy = this.pos[a].y - this.pos[b].y; return dx * dx + dy * dy; };
BrainSim.prototype.leftOf = function () { var self = this, all = []; for (var i = 0; i < this.n; i++) { var ids = []; for (var k = 0; k < this.n; k++) if (k !== i && this.pos[k].x < this.pos[i].x) ids.push(k); all.push(ids.sort(function (a, b) { return self.d2(i, a) - self.d2(i, b) || a - b; })); } return all; };
BrainSim.prototype.near = function (i) { if (this.leftList) return this.leftList[i]; return this.nearAll(i); };
BrainSim.prototype.nearAll = function (i) { var self = this, ids = []; for (var k = 0; k < this.n; k++) if (k !== i) ids.push(k); return ids.sort(function (a, b) { return self.d2(i, a) - self.d2(i, b) || a - b; }); };
BrainSim.prototype.slotOf = function (owner, other) { for (var s = 0; s < this.cap; s++) { var L = this.links[owner * this.cap + s]; if (L && L.id === other) return s; } return -1; };
BrainSim.prototype.free = function (owner) { for (var s = 0; s < this.cap; s++) if (!this.links[owner * this.cap + s]) return s; return -1; };
BrainSim.prototype.clone = function (owner, L) { for (var s = 0; s < this.cap; s++) { var C = this.links[L.id * this.cap + s]; if (C && C.id === owner && C.inward !== L.inward) return C; } return null; };
BrainSim.prototype.degree = function (o) { var d = { in: 0, out: 0 }; for (var s = 0; s < this.cap; s++) { var L = this.links[o * this.cap + s]; if (L) { if (L.inward) d.in++; else d.out++; } } return d; };
BrainSim.prototype.hops = function () { // shortest path of inward links from an input to the first output (-1: none)
  var n = this.n, dist = new Array(n).fill(-1), q = [];
  for (var i = 0; i < this.input; i++) { dist[i] = 0; q.push(i); }
  while (q.length) { var u = q.shift(); for (var v = 0; v < n; v++) { if (dist[v] >= 0) continue; for (var s = 0; s < this.cap; s++) { var L = this.links[v * this.cap + s]; if (L && L.inward && L.id === u) { dist[v] = dist[u] + 1; q.push(v); break; } } } }
  var best = -1; for (var k = n - this.output; k < n; k++) if (dist[k] >= 0 && (best < 0 || dist[k] > best)) best = dist[k]; // every output reached: the farthest one
  for (var k2 = n - this.output; k2 < n; k2++) if (dist[k2] < 0) return -1;
  return best;
};
// The inward links of each neuron packed once (rebuilt only when a link is added or removed): start[k] .. start[k + 1] in ids / slots, read in the order of the slots
BrainSim.prototype.index = function () {
  var n = this.n, cap = this.cap, start = new Int32Array(n + 1), ids = [], slots = [], links = [];
  for (var k = 0; k < n; k++) { start[k] = ids.length; for (var s = 0; s < cap; s++) { var L = this.links[k * cap + s]; if (L && L.inward) { ids.push(L.id); slots.push(k * cap + s); links.push(L); } } }
  start[n] = ids.length;
  this.inStart = start; this.inId = Int32Array.from(ids); this.inSlot = Int32Array.from(slots); this.inLink = links; this.dirty = false;
};
BrainSim.prototype.forward = function (x) {
  if (this.dirty !== false) this.index();
  var n = this.n, act = this.act, input = this.input, thr = this.threshold, start = this.inStart, ids = this.inId, links = this.inLink, bias = this.bias;
  if (!this.bufStates || this.bufStates[0].length !== n || this.bufStates.length !== this.steps + 1) { // the buffers of the steps, reused by every forward (the states stay valid until the next one)
    this.bufStates = []; this.bufSums = [null];
    for (var b = 0; b <= this.steps; b++) { this.bufStates.push(new Float64Array(n)); if (b) this.bufSums.push(new Float64Array(n)); }
  }
  var states = this.bufStates, sums = this.bufSums, s0 = states[0], seq = x.length > 0 && Array.isArray(x[0]); // a sequence: step t reads x[t], the inputs are 0 after its last step (RawBrainNetwork::sequence)
  s0.fill(0); for (var i = 0; i < input; i++) s0[i] = seq ? x[0][i] : x[i];
  for (var t = 0; t < this.steps; t++) {
    var prev = states[t], next = states[t + 1], sum_ = sums[t + 1];
    for (var i2 = 0; i2 < input; i2++) next[i2] = seq ? (t + 1 < x.length ? x[t + 1][i2] : 0) : x[i2];
    for (var k = input; k < n; k++) {
      var sum = bias[k];
      for (var q = start[k], e = start[k + 1]; q < e; q++) sum += links[q].w * prev[ids[q]];
      sum_[k] = sum;
      var v = act.f(sum); next[k] = Math.abs(v) < thr ? 0 : v;
    }
  }
  this.sums = sums;
  if (this.seen) for (var q = this.input; q < n; q++) { var y = states[states.length - 1][q]; this.seen.lo[q] = Math.min(this.seen.lo[q], y); this.seen.hi[q] = Math.max(this.seen.hi[q], y); } // revive: the range of each neuron over the round
  this.states = states; return Array.prototype.slice.call(states[states.length - 1], n - this.output);
};
BrainSim.prototype.learn = function (scores) { this.accumulate(scores); return this.apply(); }; // one sample: the gradient applied at once
// Adds the gradient of the last forward (scores: positive = good, ascent) to the pending one, apply() changes the weights (mini-batch: several accumulate, one apply)
BrainSim.prototype.accumulate = function (scores) {
  var n = this.n, cap = this.cap, grad = new Float64Array(n), act = this.act, start = this.inStart, ids = this.inId, slots = this.inSlot, links = this.inLink, direct = this.opt.direct, firstOut = n - this.output;
  if (!this.accW) { this.accW = new Float64Array(n * cap); this.accB = new Float64Array(n); }
  var bg = this.accB, wg = this.accW;
  if (!this.bufGrad || this.bufGrad.length !== n) { this.bufGrad = new Float64Array(n); this.bufBelow = new Float64Array(n); }
  grad = this.bufGrad; grad.fill(0); var spare = this.bufBelow;
  for (var k = 0; k < this.output; k++) grad[n - this.output + k] = scores[k];
  for (var t = this.steps; t > 0; t--) {
    var before = this.states[t - 1], sums = this.sums[t], below = spare; below.fill(0);
    for (var i = this.input; i < n; i++) {
      if (grad[i] === 0) continue;
      var sum = sums[i];
      var slope = t === this.steps && i >= firstOut && direct ? grad[i] : grad[i] * act.d(sum, act.f(sum)); // direct: the score reaches the sum of an output as it is (cross-entropy form: a saturated output still learns)
      bg[i] += slope;
      for (var q = start[i], e = start[i + 1]; q < e; q++) { wg[slots[q]] += slope * before[ids[q]]; below[ids[q]] += slope * links[q].w; }
    }
    spare = grad; grad = below;
  }
};
// Momentum: step = momentum * last step + gradient; stepClip: each step kept in [-clip, clip] (0 = no limit)
BrainSim.prototype.apply = function () {
  var n = this.n, cap = this.cap, m = this.opt.momentum || 0, clip = this.opt.stepClip || 0, bg = this.accB, wg = this.accW, events = { added: [], removed: [], changed: [] };
  var bound = this.opt.clamp ? this.max : Infinity; // clamp: |weight| and |bias| never above weightMax
  var keep = function (v) { return Math.max(-bound, Math.min(bound, v)); };
  var step = function (v) { return clip > 0 ? Math.max(-clip, Math.min(clip, v)) : v; };
  if (!bg) return events;
  if (!this.velW) { this.velW = new Array(n * cap).fill(0); this.velB = new Array(n).fill(0); }
  // Adam: m = b1 m + (1 - b1) g, v = b2 v + (1 - b2) g², step = rate * m^ / (sqrt(v^) + eps) with the corrections of the first steps (the score only gives the direction)
  var adam = this.opt.adam, b1 = 0.9, b2 = 0.999, lr = (this.opt.adamRate || 0.02) * (this.lrScale === undefined ? 1 : this.lrScale), t = 0, ams = this.opt.amsgrad; // lrScale: the decay of the round; amsgrad: v never decreases
  if (adam) { if (!this.adM) { this.adM = new Array(n * cap + n).fill(0); this.adV = new Array(n * cap + n).fill(0); this.adT = 0; } t = ++this.adT; }
  var c1 = adam ? 1 - Math.pow(b1, t) : 1, c2 = adam ? 1 - Math.pow(b2, t) : 1; // the corrections of the first steps, once per update
  var self = this, change = function (at, g, vel) {
    if (!adam) return step(m * vel + g);
    self.adM[at] = b1 * self.adM[at] + (1 - b1) * g; var v2 = b2 * self.adV[at] + (1 - b2) * g * g; self.adV[at] = ams ? Math.max(self.adV[at], v2) : v2;
    return step(lr * (self.adM[at] / c1) / (Math.sqrt(self.adV[at] / c2) + 1e-8));
  };
  for (var i3 = this.input; i3 < n; i3++) {
    if (isFinite(bg[i3])) { this.velB[i3] = change(n * cap + i3, bg[i3], this.velB[i3]); this.bias[i3] = keep(this.bias[i3] + this.velB[i3]); }
    for (var s3 = 0; s3 < cap; s3++) {
      var at = i3 * cap + s3, L3 = this.links[at];
      if (!L3 || !L3.inward) { this.velW[at] = 0; continue; }
      if (!isFinite(wg[at]) || (wg[at] === 0 && this.velW[at] === 0)) continue;
      var d = this.velW[at] = change(at, wg[at], this.velW[at]);
      L3.w = keep(L3.w + d); if (L3.w === 0) L3.w = 1e-38; this.clone(i3, L3).w = L3.w; events.changed.push([L3.id, i3, d]);
    }
  }
  this.accW = null; this.accB = null;
  if (this.relinkMode !== 'off') this.relink(events);
  return events;
};
// Revive: a neuron stuck at 0 or 1 on every sample of the round (saturated, no gradient) gets its bias back to 0 and its inward weights halved
BrainSim.prototype.watch = function () { this.seen = { lo: new Array(this.n).fill(Infinity), hi: new Array(this.n).fill(-Infinity) }; };
BrainSim.prototype.revive = function () {
  var seen = this.seen, count = 0; this.seen = null; if (!seen) return 0;
  for (var k = this.input; k < this.n; k++) {
    var stuckLow = seen.hi[k] < 0.02, stuckHigh = seen.lo[k] > 0.98;
    if (!(stuckLow || stuckHigh) || seen.hi[k] - seen.lo[k] > 1e-3) continue;
    this.bias[k] = 0; count++;
    for (var s = 0; s < this.cap; s++) { var L = this.links[k * this.cap + s]; if (L && L.inward) { L.w = L.w / 2 || 1e-38; this.clone(k, L).w = L.w; } }
  }
  return count;
};
BrainSim.prototype.removeLink = function (owner, slot) { this.dirty = true; var cur = this.links[owner * this.cap + slot], C = this.clone(owner, cur); this.links[cur.id * this.cap + this.links.slice(cur.id * this.cap, cur.id * this.cap + this.cap).indexOf(C)] = null; this.links[owner * this.cap + slot] = null; return { cur: cur, C: C }; };
BrainSim.prototype.relink = function (events) {
  var n = this.n, cap = this.cap, self = this, min = this.min, max = this.max, inverted = this.relinkMode === 'inverted';
  for (var owner = 0; owner < n; owner++) {
    var list = [], any = false;
    for (var s0 = 0; s0 < cap && !any; s0++) { var L0 = this.links[owner * cap + s0]; if (L0) { var a0 = Math.abs(L0.w); any = a0 <= min || a0 >= max; } }
    if (!any) continue; // no link at a limit: nothing changes for this neuron
    for (var s = 0; s < cap; s++) { var L = this.links[owner * cap + s]; if (L) list.push({ slot: s, id: L.id, d: this.d2(owner, L.id) }); }
    list.sort(function (a, b) { return a.d - b.d || a.id - b.id; });
    // -1 = cut, 1 = grow, 0 = nothing (inverted: cut at the max, grow at the min)
    var snap = list.map(function (e) { var w = Math.abs(self.links[owner * cap + e.slot].w), low = w <= min, high = w >= max; return { slot: e.slot, id: e.id, w: w, kind: low ? (inverted ? 1 : -1) : (high ? (inverted ? -1 : 1) : 0) }; });
    for (var k = 0; k < snap.length; k++) {
      var e = snap[k], cur = this.links[owner * cap + e.slot];
      if (!cur || cur.id !== e.id || e.kind === 0) continue;
      if (e.kind < 0) {
        var a = this.degree(owner), b = this.degree(cur.id);
        if (a.in + a.out <= this.minLinks || b.in + b.out <= this.minLinks) continue; // the forced minimum of links
        if (!this.opt.orphans && !(this.survives(a, owner < this.input, cur.inward) && this.survives(b, cur.id < this.input, !cur.inward))) continue;
        var reached = this.opt.protectPath ? this.pairs() : 0, removed = this.removeLink(owner, e.slot);
        if (this.opt.protectPath && this.pairs() < reached) { // the last path of an input to an output: put it back
          this.links[owner * cap + e.slot] = removed.cur; this.links[removed.cur.id * cap + this.free(removed.cur.id)] = removed.C; continue;
        }
        events.removed.push(cur.inward ? [cur.id, owner] : [owner, cur.id]);
        continue;
      }
      if (this.free(owner) < 0) continue;
      var ref = 0, used = 0;
      for (var dist = 1; dist < snap.length && used < 2; dist++) for (var side = -1; side <= 1; side += 2) { var at = k + side * dist; if (at < 0 || at >= snap.length || used >= 2 || snap[at].kind !== 0) continue; ref += snap[at].w; used++; }
      ref = used === 0 ? (min + max) / 2 : ref / used;
      var order = this.near(owner);
      for (var q = 0; q < order.length; q++) {
        var other = order[q];
        if (this.slotOf(owner, other) >= 0 || this.free(other) < 0) continue;
        var dir = this.direction(owner, other); if (dir === null) continue;
        var low2 = min + (max - min) * 0.05, high2 = min + (max - min) * 0.95;
        var value = (this.r() < 0.5 ? -1 : 1) * (this.opt.newLink === 'low' ? low2 : Math.min(Math.max(ref * (0.5 + 0.5 * this.r()), low2), high2)); // low: a small weight, no shock on the outputs
        if (this.velW) { this.velW[owner * cap + this.free(owner)] = 0; this.velW[other * cap + this.free(other)] = 0; }
        if (this.adM) [owner * cap + this.free(owner), other * cap + this.free(other)].forEach(function (k) { self.adM[k] = 0; self.adV[k] = 0; }); // a new link starts from rest
        this.dirty = true;
        this.links[owner * cap + this.free(owner)] = { id: other, w: value, inward: dir };
        this.links[other * cap + this.free(other)] = { id: owner, w: value, inward: !dir };
        events.added.push(dir ? [other, owner] : [owner, other]);
        break;
      }
    }
  }
};
BrainSim.prototype.survives = function (d, isInput, inward) { var i = d.in - (inward ? 1 : 0), o = d.out - (inward ? 0 : 1), before = isInput ? d.out : d.in, after = isInput ? o : i; return i + o >= 1 && !(before >= 1 && after === 0); };
// Caps of the library: min(floor(percentage * n), hook(n)) (links: clamped to [1, n - 1]; steps: at least 1)
function linksCapOf(n, pct, hook) { if (n < 2) return 0; var h = hook === null || hook === undefined ? Math.floor(3 * Math.sqrt(n)) : hook; return Math.min(Math.max(Math.min(Math.floor(n * pct), h), 1), n - 1); }
function stepsCapOf(n, pct, hook) { var h = hook === null || hook === undefined ? Math.floor(Math.log2(n)) : hook; return Math.max(1, Math.min(Math.floor(n * pct), h)); }
if (typeof module !== 'undefined') module.exports = { LayerSim: LayerSim, BrainSim: BrainSim, rng: rng, ACT: ACT, linksCapOf: linksCapOf, stepsCapOf: stepsCapOf };
