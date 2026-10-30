// Lab: build both networks from the form, train them round by round on the task, draw them and compare
document.addEventListener('DOMContentLoaded', function () {
  var $ = function (id) { return document.getElementById(id); };
  var state = { task: null, layer: null, brain: null, round: 0, lh: [], bh: [], lth: [], bth: [], added: 0, removed: 0, running: false, timer: null, lt: 0, bt: 0, engine: 'js', remote: null, busy: false, creating: false, retry: null, build: 0, custom: null, pad: null, preset: false }; // engine: 'js' (the page) | 'server' (the C++ library, remote = id of its network), build: counter (an old answer is dropped) // *h: error per round (t: on the test samples), lt / bt: training time (ms)
  function num(id, fallback) { var v = parseFloat($(id).value); return isFinite(v) ? v : fallback; }
  function optNum(id) { var v = $(id).value.trim(); return v === '' ? null : parseFloat(v); }
  function ms(v) { return v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v.toFixed(1) + ' ms'; }
  function f2(v) { return isFinite(v) ? (Math.round(v * 100) / 100).toFixed(2) : String(v); }
  function vec(v) { return v.length === 1 ? f2(v[0]) : v.map(f2).join(' '); }
  function best(v) { var k = 0; for (var i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; }
  function outText(t, v) { return t.classes && v.length > 1 ? t.classes[best(v)] + ' ' + f2(v[best(v)]) : vec(v); } // a class: its name and its output
  function sampleText(t, i) { var x = t.all.in[i]; return (t.all.label ? t.all.label[i] : '(' + x.map(f2).join(', ') + ')') + ' → ' + outText(t, t.all.out[i]); }
  function taskOpt() { return { density: num('density', 17), density2d: num('density2d', 9), noise: num('noise', 0), test: num('testPct', 0), seed: Math.round(num('dataSeed', 1)) }; }
  function hidden() { return Array.prototype.map.call(document.querySelectorAll('#l-list input'), function (el) { return Math.max(1, parseInt(el.value, 10) || 1); }); } // neurons of each hidden layer
  function layerField(value) {
    var label = document.createElement('label'), input = document.createElement('input');
    input.type = 'number'; input.min = '1'; input.value = value;
    label.textContent = 'Layer ' + ($('l-list').children.length + 1); label.appendChild(input);
    input.addEventListener('change', build); // the network is built again with its seed, Play keeps running
    $('l-list').appendChild(label);
  }
  function brainOpt(seed) {
    var t = state.task, n = Math.max(0, Math.round(num('b-neurons', 13))) + t.inputs + t.outputs; // the hidden neurons, plus the inputs and the outputs of the task (the library counts them all)
    return { seed: seed, neurons: n, input: t.inputs, output: t.outputs, width: num('b-width', 8), height: num('b-height', 5),
      cap: linksCapOf(n, num('b-lpct', 1), optNum('b-lhook')), steps: stepsCapOf(n, num('b-spct', 1), optNum('b-shook')),
      min: num('b-wmin', 0.01), max: num('b-wmax', 5), range: num('b-range', 3), threshold: num('b-thr', 0), activation: $('b-act').value,
      weightInit: $('b-wi').checked, biasInit: $('b-bi').checked, linkInit: $('b-linit').value, minLinks: Math.max(0, Math.round(num('b-minlinks', 0))),
      direction: $('b-dir').value, relink: $('b-relink').value, protectPath: $('b-protect').checked, orphans: $('b-orphans').checked,
      momentum: num('b-mom', 0), stepClip: num('b-clip', 0), newLink: $('b-newlink').value, longLinks: Math.round(num('b-long', 0)), pathInit: $('b-path').checked, pathHops: Math.round(num('b-pathhops', 3)), clamp: $('b-clamp').checked, direct: $('b-direct').checked,
      adam: $('b-adam').checked, adamRate: num('b-adamrate', 0.01), revive: $('b-revive').checked };
  }
  function brainTrain() { return { rate: num('b-rate', 3), coef: $('b-score').value, decay: num('b-decay', 0), batch: $('b-batch').checked }; }
  // The options of the server: the library names of the same settings
  function serverOptions(seed) {
    var o = brainOpt(seed), tr = brainTrain();
    return { seed: seed, neurons: o.neurons, input: o.input, output: o.output, width: o.width, height: o.height, linksPct: num('b-lpct', 1), linksHook: optNum('b-lhook'), stepsPct: num('b-spct', 1), stepsHook: optNum('b-shook'),
      weightMin: o.min, weightMax: o.max, range: o.range, threshold: o.threshold, activation: o.activation, weightInit: o.weightInit, biasInit: o.biasInit, direction: o.direction, relink: o.relink,
      keepPaths: o.protectPath, orphans: o.orphans, newLink: o.newLink, farLinks: o.longLinks, pathInit: o.pathInit, pathHops: o.pathHops, momentum: o.momentum, adam: o.adam, adamRate: o.adamRate,
      stepLimit: o.stepClip, clamp: o.clamp, direct: o.direct, batch: tr.batch, linkInit: o.linkInit, minLinks: o.minLinks, filter: Math.round(num('b-filter', 0)), allInputs: $('b-allin').checked, gpu: $('engineGpu').checked };
  }
  function serverTask(t) { return { in: t.in, out: t.out, clean: t.clean, testIn: t.test.in, testOut: t.test.out, kind: t.kind }; }
  // The points the page reads on the brain of the server: every sample, then the curve (101 points) or the map (30 x 30)
  function points(t) {
    var p = t.all.in.slice();
    if (state.custom) p.push(state.custom); // the input of the user (the panel)
    if (t.dims === 1) for (var k = 0; k <= 100; k++) p.push([k / 100]);
    if (t.dims === 2) for (var a = 0; a < 30; a++) for (var c = 0; c < 30; c++) p.push([(a + 0.5) / 30, (c + 0.5) / 30]);
    return p;
  }
  // The server engine stays chosen when it stops answering: it is asked again every 2 s, the network is built again once it is back
  function engineLost(e) {
    state.remote = null; state.creating = false; state.busy = false;
    $('msg').textContent = 'Server engine not answering (' + (e && e.message ? e.message : e) + '): trying again every 2 s (lab/server.sh; choose JavaScript in Engine to run in the page).';
    if (state.retry) return;
    state.retry = setTimeout(function retry() {
      LAB_SERVER.info(1500).then(function () { state.retry = null; if (state.engine === 'server') build(); }).catch(function () { state.retry = setTimeout(retry, 2000); });
    }, 2000);
  }
  function engineFail(e, id) { // an answer for an older network is dropped; the server down: engineLost; an error of the server: said, the network built again
    if (id !== undefined && id !== state.build) return;
    if (e && e.network) { engineLost(e); return; }
    state.remote = null; state.creating = false; state.busy = false;
    $('msg').textContent = 'Server engine error (' + (e && e.message ? e.message : e) + '): the network is built again.';
    setTimeout(build, 300);
  }
  function remoteUse(data, t, pts, sample) { // the mirror of the answer, the seed of the server, what it ignored
    state.brain = new RemoteBrain(data.snapshot || data, pts, brainOpt(1), sample);
    var snap = data.snapshot || data;
    if (snap.seed !== undefined) $('b-seed').value = snap.seed;
    $('msg').textContent = 'Server engine' + (snap.gpu ? ' on ' + snap.gpu : ' (host)') + (snap.unsupported && snap.unsupported.length ? ' — ignored by the library: ' + snap.unsupported.join(', ') : '');
  }
  function brainHops(b) { return $('b-allin').checked ? b.hopsAll() : b.hops(); }
  function record() { var t = state.task; state.lh.push(labError(state.layer, t.in, t.clean)); state.bh.push(labError(state.brain, t.in, t.clean)); state.lth.push(labError(state.layer, t.test.in, t.test.out)); state.bth.push(labError(state.brain, t.test.in, t.test.out)); }
  function build() {
    state.task = makeTask($('task').value, taskOpt());
    var t = state.task;
    $('taskInfo').textContent = t.name + ': ' + LAB_TASK_INFO[t.name] + ' — ' + (t.seq ? '1 input per time step (' + t.all.in[0].length + ' steps; the layered network gets ' + t.layerInputs + ' inputs)' : t.inputs + ' input(s)') + ', ' + t.outputs + ' output(s), ' + t.in.length + ' trained sample(s)' + (t.test.in.length ? ', ' + t.test.in.length + ' for the test' : '') + (t.kind === 'reg' ? ', regression' : '');
    if (num('b-wmin', 0) >= num('b-wmax', 1)) $('msg').textContent = 'weightMin should be under weightMax: the library refuses it.'; else $('msg').textContent = '';
    padBuild();
    state.layer = new LayerSim([t.layerInputs].concat(hidden(), [t.outputs]), Math.round(num('l-seed', 3)), { activation: $('l-act').value, range: num('l-range', 1), weightInit: $('l-wi').checked, biasInit: $('l-bi').checked });
    state.brain = new BrainSim(brainOpt(Math.round(num('b-seed', 1)))); // the server one replaces it when it answers
    state.round = 0; state.added = 0; state.removed = 0; state.lt = 0; state.bt = 0; state.lh = []; state.bh = []; state.lth = []; state.bth = []; record();
    var sel = $('sample'), keep = parseInt(sel.value, 10);
    sel.innerHTML = t.all.in.map(function (x, i) { return '<option value="' + i + '">' + (t.isTest[i] ? '[test] ' : '') + sampleText(t, i) + '</option>'; }).join('');
    sel.value = String(keep >= 0 && keep < t.all.in.length ? keep : 0);
    draw();
    if (state.engine !== 'server') return;
    var pts = points(t), sample = shownSample(), id = ++state.build, previous = state.remote;
    state.creating = true; state.remote = null; // no training of the old network meanwhile
    LAB_SERVER.post('/api/create', { options: serverOptions(Math.round(num('b-seed', 1))), task: serverTask(t), points: pts, sample: sample, previous: previous }).then(function (data) {
      if (id !== state.build) return; // built again meanwhile
      state.remote = data.id; state.creating = false; remoteUse(data, t, pts, sample);
      state.bh = [labError(state.brain, t.in, t.clean)]; state.bth = [labError(state.brain, t.test.in, t.test.out)];
      draw();
    }).catch(function (e) { engineFail(e, id); });
  }
  // Rounds of the server engine: the layered network trains on the page meanwhile, the brain answers with its errors of each round
  // budget (ms): Play asks the rounds that fit in it (the server sends a frame at least that often), 0 = exactly <rounds> (+1 / +100)
  function serverRounds(rounds, budget) {
    if (state.busy || state.creating || state.remote === null) return Promise.resolve(false);
    var t = state.task, pts = points(t), sample = shownSample(), id = state.build, lr = num('l-rate', 1.5), lc = $('l-coef').value;
    state.busy = true;
    return LAB_SERVER.post('/api/train', { id: state.remote, rounds: rounds, budget: budget || 0, train: Object.assign(brainTrain(), { revive: $('b-revive').checked }), sample: sample }).then(function (data) {
      state.busy = false;
      if (id !== state.build) return false;
      var done = data.errors.length, t0 = performance.now(); // the layered network trains as many rounds on the page
      for (var r = 0; r < done; r++) { for (var i = 0; i < t.in.length; i++) state.layer.learn(labLayerCoef(lc, lr, state.layer.forward(t.in[i]), t.out[i])); state.lh.push(labError(state.layer, t.in, t.clean)); state.lth.push(labError(state.layer, t.test.in, t.test.out)); }
      state.lt += performance.now() - t0;
      remoteUse(data, t, pts, sample);
      state.bh = state.bh.concat(data.errors.map(function (v) { return v === null ? NaN : v; })); state.bth = state.bth.concat(data.testErrors.map(function (v) { return v === null ? NaN : v; }));
      state.bt += data.ms; state.round += done;
      return true;
    }).catch(function (e) { state.busy = false; engineFail(e, id); return false; });
  }
  function regenerate() {
    if ($('randomSeeds').checked) {
      $('l-seed').value = 1 + Math.floor(Math.random() * 1e9);
      var filter = Math.round(num('b-filter', 0)), seed;
      for (var tries = 0; tries < 500; tries++) { seed = 1 + Math.floor(Math.random() * 1e9); if (!filter || state.engine === 'server') break; var b = new BrainSim(brainOpt(seed)), h = brainHops(b); if (h >= 1 && h <= filter) break; } // the server filters its own networks
      $('b-seed').value = seed;
    }
    build();
  }
  function trainRound() {
    var t = state.task, lr = num('l-rate', 1.5), lc = $('l-coef').value, i;
    var t0 = performance.now();
    for (i = 0; i < t.in.length; i++) state.layer.learn(labLayerCoef(lc, lr, state.layer.forward(t.in[i]), t.out[i]));
    var t1 = performance.now();
    var ev = labBrainRound(state.brain, t, brainTrain(), state.round); state.added += ev.added; state.removed += ev.removed;
    state.lt += t1 - t0; state.bt += performance.now() - t1;
    state.round++; record();
  }
  function done() { var maxRounds = num('maxRounds', 0), stopErr = num('stopErr', 0); return (maxRounds > 0 && state.round >= maxRounds) || (stopErr > 0 && state.lh[state.lh.length - 1] < stopErr && state.bh[state.bh.length - 1] < stopErr); } // the training holds, Play stays on
  function loop() { // only a click on Play stops it
    state.timer = null;
    if (state.engine === 'server') {
      if (done() || state.creating || state.remote === null) { if (state.running) state.timer = setTimeout(loop, 200); return; } // held, or the network of the server not there yet
      var hold = num('maxRounds', 0), left = hold > 0 ? Math.max(1, hold - state.round) : 100000;
      serverRounds(left, 40).then(function () { draw(); if (state.running && !state.timer) state.timer = setTimeout(loop, 0); }); // as many rounds as 40 ms allow, then a frame
      return;
    }
    for (var r = 0; r < Math.max(1, num('rpf', 5)) && !done(); r++) trainRound(); draw(); if (state.running) state.timer = setTimeout(loop, 30);
  }
  function setPlay(on) { state.running = on; var b = document.querySelector('[data-act="play"]'); b.textContent = on ? 'Pause' : 'Play'; b.setAttribute('aria-pressed', on ? 'true' : 'false'); if (on && !state.timer) loop(); }

  // ---------- drawing ----------
  function edge(cls, x1, y1, x2, y2, w, marker) { return '<line class="vz-edge ' + cls + '" x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke-width="' + w.toFixed(2) + '"' + (marker ? ' marker-end="url(#' + marker + ')"' : '') + '/>'; }
  function node(x, y, r, v, bias, io) {
    var s = '<circle class="vz-node' + (io ? ' vz-io' : '') + '" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r + '"/>';
    if (v !== null) s += '<circle class="vz-fill" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r + '" fill-opacity="' + (0.12 + 0.85 * Math.min(Math.abs(v), 1)).toFixed(2) + '"/>';
    if (bias !== null) s += '<circle class="vz-ring ' + (bias >= 0 ? 'vz-pos' : 'vz-neg') + '" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (r + 3) + '" stroke-opacity="' + Math.min(1, 0.2 + Math.abs(bias) / 3).toFixed(2) + '"/>';
    if (v !== null && r >= 11) s += '<text class="vz-val' + (Math.abs(v) > 0.55 ? ' vz-on' : '') + '" font-size="' + (r >= 18 ? 12 : 10) + '" x="' + x.toFixed(1) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="middle">' + (Math.round(v * 100) / 100).toFixed(r >= 18 ? 2 : 1) + '</text>';
    return s;
  }
  function shownSample() { var t = state.task, i = parseInt($('sample').value, 10) || 0; return state.custom && $('inShow').checked ? state.custom : t.all.in[Math.min(i, t.all.in.length - 1)]; }
  function testStats(h) { return state.task.test.in.length ? '<span>Test error <b>' + f2(h[h.length - 1]) + '</b></span>' : ''; }
  function drawLayer() {
    var net = state.layer, t = state.task, sizes = net.sizes, Lc = sizes.length, maxN = Math.max.apply(null, sizes);
    net.forward(shownSample()); var outs = net.last.outs, r = Math.max(5, Math.min(22, Math.floor(130 / maxN))), s = '';
    var X = function (l) { return 70 + l * 620 / (Lc - 1); }, Y = function (l, k) { return 30 + (k + 0.5) * 310 / sizes[l]; };
    for (var l = 0; l < net.layers.length; l++) net.layers[l].forEach(function (n, k) { n.w.forEach(function (w, j) { s += edge(w >= 0 ? 'vz-pos' : 'vz-neg', X(l), Y(l, j), X(l + 1), Y(l + 1, k), 0.6 + Math.min(Math.abs(w), 6) * 0.8); }); });
    for (var l2 = 0; l2 < Lc; l2++) for (var k2 = 0; k2 < sizes[l2]; k2++) s += node(X(l2), Y(l2, k2), r, outs[l2][k2], l2 > 0 ? net.layers[l2 - 1][k2].b : null, l2 === 0 || l2 === Lc - 1);
    $('l-svg').innerHTML = s;
    var weights = 0; net.layers.forEach(function (layer) { layer.forEach(function (n) { weights += n.w.length; }); });
    $('l-info').textContent = 'layers {' + sizes.slice(1).join(', ') + '}, ' + weights + ' weights';
    $('l-stats').innerHTML = '<span>Seed <b>' + $('l-seed').value + '</b></span><span>Round <b>' + state.round + '</b></span><span>Mean error <b>' + f2(state.lh[state.lh.length - 1]) + '</b></span>' + testStats(state.lth) + '<span>Solved <b>' + (labSolved(t, net, t.in, t.clean) ? 'yes' : 'no') + '</b></span><span>Train time <b>' + ms(state.lt) + '</b></span>';
  }
  function drawBrain() {
    var net = state.brain, t = state.task, n = net.n, cap = net.cap, W = net.opt.width, H = net.opt.height;
    net.forward(shownSample()); var st = parseInt($('step').value, 10), states = net.states, at = isFinite(st) && st >= 0 ? Math.min(st, states.length - 1) : states.length - 1, state0 = states[at];
    var r = n > 120 ? 5 : (n > 50 ? 8 : 14), px = function (k) { return 60 + net.pos[k].x / W * 640; }, py = function (k) { return 25 + net.pos[k].y / H * 310; }, links = 0;
    var s = '<defs><marker id="labArr" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="' + (r > 8 ? 9 : 6) + '" markerHeight="' + (r > 8 ? 9 : 6) + '" orient="auto"><path class="arrow" d="M0 0L10 5L0 10z"/></marker></defs>';
    for (var k = 0; k < n; k++) for (var q = 0; q < cap; q++) {
      var L = net.links[k * cap + q]; if (!L || !L.inward) continue; links++;
      var x1 = px(L.id), y1 = py(L.id), x2 = px(k), y2 = py(k), len = Math.hypot(x2 - x1, y2 - y1) || 1, ux = (x2 - x1) / len, uy = (y2 - y1) / len;
      var hot = at > 0 && Math.abs(L.w * states[at - 1][L.id]) > 0.05;
      s += edge((L.w >= 0 ? 'vz-pos' : 'vz-neg') + (hot ? ' vz-hot' : ''), x1 + ux * (r + 2), y1 + uy * (r + 2), x2 - ux * (r + 5), y2 - uy * (r + 5), 0.6 + Math.min(Math.abs(L.w), 4) * (r > 8 ? 1 : 0.5), 'labArr');
    }
    for (var j = 0; j < n; j++) s += node(px(j), py(j), r, state0[j], j >= net.input ? net.bias[j] : null, j < net.input || j >= n - net.output);
    $('b-svg').innerHTML = s;
    var h = net.hops(), ha = net.hopsAll();
    $('b-info').textContent = n + ' neurons (' + (n - net.input - net.output) + ' hidden + ' + net.input + ' in + ' + net.output + ' out), cap ' + cap + ', ' + net.steps + ' steps, ' + links + ' links (step shown ' + at + ')';
    $('b-stats').innerHTML = '<span>Seed <b>' + $('b-seed').value + '</b></span><span>Round <b>' + state.round + '</b></span><span>Mean error <b>' + f2(state.bh[state.bh.length - 1]) + '</b></span>' + testStats(state.bth) + '<span>Solved <b>' + (labSolved(t, net, t.in, t.clean) ? 'yes' : 'no') + '</b></span><span>Path input → outputs <b>' + (h < 0 ? 'none' : h + ' hop(s)' + (h > net.steps ? ' > steps' : '')) + '</b></span><span>Every input <b>' + (ha < 0 ? 'no' : ha + ' hop(s)' + (ha > net.steps ? ' > steps' : '')) + '</b></span><span>Links <b>' + (net.remote ? 'n/a (server)' : '+' + state.added + ' / −' + state.removed) + '</b></span><span>Train time <b>' + ms(state.bt) + '</b></span>';
  }
  function drawChart(log) { // log: the same curves on a log scale (the small errors and the jumps of the end can be read)
    var Wd = 760, Hd = 260, l = 50, rr = 15, tp = 12, b = 30, all = state.lh.concat(state.bh, state.lth, state.bth).filter(isFinite), ymax = Math.min(1.5, Math.max.apply(null, all.concat([0.1])));
    var positive = all.filter(function (v) { return v > 0; }), top = Math.ceil(Math.log10(Math.max.apply(null, positive.concat([1e-6])))), bottom = Math.max(-6, Math.floor(Math.log10(Math.min.apply(null, positive.concat([1]))))); // decades shown: from the smallest error to the largest
    if (bottom >= top) bottom = top - 1;
    var len = Math.max(state.lh.length - 1, 1), s = '';
    var x = function (i) { return l + (Wd - l - rr) * i / len; }, y = log ? function (v) { var e = Math.log10(Math.max(v, Math.pow(10, bottom))); return tp + (Hd - tp - b) * (top - Math.min(e, top)) / (top - bottom); } : function (v) { return tp + (Hd - tp - b) * (1 - Math.min(v, ymax) / ymax); };
    var ticks = log ? (function () { var t = []; for (var d = bottom; d <= top; d++) t.push(Math.pow(10, d)); return t; })() : [0, 1, 2, 3, 4].map(function (g) { return ymax * g / 4; });
    ticks.forEach(function (v) { s += '<line class="grid" x1="' + l + '" x2="' + (Wd - rr) + '" y1="' + y(v) + '" y2="' + y(v) + '"/><text x="' + (l - 6) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + (log ? (v >= 0.01 ? String(+v.toFixed(2)) : v.toExponential(0)) : v.toFixed(2)) + '</text>'; });
    s += '<line class="axis" x1="' + l + '" x2="' + l + '" y1="' + tp + '" y2="' + (Hd - b) + '"/><text x="' + (Wd - rr) + '" y="' + (Hd - 8) + '" text-anchor="end">round ' + state.round + '</text><text x="' + l + '" y="' + (Hd - 8) + '">0</text>';
    var step = Math.max(1, Math.floor(state.lh.length / 600));
    [['lh', 'c-layer'], ['bh', 'c-brain'], ['lth', 'c-layer c-test'], ['bth', 'c-brain c-test']].forEach(function (p) {
      var pts = []; for (var i = 0; i < state[p[0]].length; i += step) if (isFinite(state[p[0]][i])) pts.push(x(i).toFixed(1) + ',' + y(state[p[0]][i]).toFixed(1));
      if (pts.length) s += '<polyline class="' + p[1] + '" points="' + pts.join(' ') + '"/>';
    });
    $(log ? 'chartLog' : 'chart').innerHTML = s;
  }
  // 1D: the samples and both predicted curves; 2D: a map of the first output of each network with the samples on it
  function drawFunction() {
    var t = state.task, card = $('fnCard');
    card.style.display = t.dims ? '' : 'none'; if (!t.dims) return;
    $('fn-info').textContent = t.dims === 1 ? 'the predicted curve of each network on [0, 1]' : 'first output of each network on the square [0, 1]² (dark = 1)';
    var dots = function (X, Y) { return t.all.in.map(function (p, i) { return t.isTest[i] ? '<circle class="dot-test" cx="' + X(p).toFixed(1) + '" cy="' + Y(p, i).toFixed(1) + '" r="3.5"/>' : '<circle class="dot" cx="' + X(p).toFixed(1) + '" cy="' + Y(p, i).toFixed(1) + '" r="3"/>'; }).join(''); };
    if (t.dims === 1) {
      var W = 760, H = 260, l = 40, rr = 15, tp = 10, b = 26, xs = function (v) { return l + (W - l - rr) * v; }, ys = function (v) { return tp + (H - tp - b) * (1 - Math.max(-0.1, Math.min(1.1, v)) / 1.0); }, s = '';
      for (var g = 0; g <= 4; g++) s += '<line class="grid" x1="' + l + '" x2="' + (W - rr) + '" y1="' + ys(g / 4) + '" y2="' + ys(g / 4) + '"/><text x="' + (l - 6) + '" y="' + (ys(g / 4) + 4) + '" text-anchor="end">' + (g / 4).toFixed(2) + '</text>';
      [[state.layer, 'c-layer'], [state.brain, 'c-brain']].forEach(function (p) { var pts = []; for (var k = 0; k <= 100; k++) { var v = k / 100; pts.push(xs(v).toFixed(1) + ',' + ys(p[0].forward([v])[0]).toFixed(1)); } s += '<polyline class="' + p[1] + '" points="' + pts.join(' ') + '"/>'; });
      s += dots(function (p) { return xs(p[0]); }, function (p, i) { return ys(t.all.out[i][0]); });
      $('fn').innerHTML = '<svg viewBox="0 0 760 260" role="img" aria-label="Predicted curves">' + s + '</svg>';
      return;
    }
    var cells = 30, html = '';
    [[state.layer, 'Layered'], [state.brain, 'Brain']].forEach(function (p) {
      var S = 300, m = 24, sz = S / cells, s = '<text class="map-title" x="' + m + '" y="16">' + p[1] + '</text>';
      for (var a = 0; a < cells; a++) for (var c = 0; c < cells; c++) {
        var v = Math.max(0, Math.min(1, p[0].forward([(a + 0.5) / cells, (c + 0.5) / cells])[0]));
        s += '<rect x="' + (m + a * sz).toFixed(1) + '" y="' + (m + (cells - 1 - c) * sz).toFixed(1) + '" width="' + (sz + 0.3).toFixed(2) + '" height="' + (sz + 0.3).toFixed(2) + '" style="fill:var(--accent)" fill-opacity="' + v.toFixed(2) + '"/>';
      }
      s += dots(function (q) { return m + q[0] * S; }, function (q) { return m + (1 - q[1]) * S; });
      s += t.all.in.map(function (q, i) { return '<circle cx="' + (m + q[0] * S).toFixed(1) + '" cy="' + (m + (1 - q[1]) * S).toFixed(1) + '" r="1.6" style="fill:' + (t.all.out[i][0] >= 0.5 ? 'var(--ok)' : 'var(--warn)') + '"/>'; }).join('');
      html += '<svg viewBox="0 0 ' + (S + 2 * m) + ' ' + (S + 2 * m) + '" role="img" aria-label="' + p[1] + ' prediction map">' + s + '</svg>';
    });
    $('fn').innerHTML = html;
  }
  function drawOutputs() {
    var t = state.task, shown = parseInt($('sample').value, 10) || 0, s = '<thead><tr><th>Sample</th><th>Expected</th><th>Layered</th><th>Brain</th></tr></thead><tbody>', max = 64;
    t.all.in.slice(0, max).forEach(function (x, i) {
      var a = state.layer.forward(x), c = state.brain.forward(x), want = t.all.out[i];
      var label = t.all.label ? t.all.label[i] : '(' + x.map(f2).join(', ') + ')';
      s += '<tr class="' + (i === shown ? 'shown' : '') + (t.isTest[i] ? ' test' : '') + '"><td>' + (t.isTest[i] ? 'test ' : '') + label + '</td><td>' + outText(t, want) + '</td><td class="' + (labGood(t, a, want) ? 'ok' : 'bad') + '">' + outText(t, a) + '</td><td class="' + (labGood(t, c, want) ? 'ok' : 'bad') + '">' + outText(t, c) + '</td></tr>';
    });
    if (t.all.in.length > max) s += '<tr><td colspan="4">… ' + (t.all.in.length - max) + ' more samples (pick one in "Sample shown")</td></tr>';
    $('outs').innerHTML = s + '</tbody>';
  }
  function draw() { drawLayer(); drawBrain(); drawChart(false); drawChart(true); drawFunction(); drawOutputs(); drawInput(); }

  // ---------- the input of the user (scenarios): a digit drawn, a word typed, a sequence of bits, a board ----------
  // state.pad: what the panel holds (pixels / word / bits / board), state.custom: its inputs for the networks (null = no panel)
  function padEncode() {
    var t = state.task, p = state.pad;
    if (!p) return null;
    if (t.name === 'DIGITS') return p.slice();
    if (t.name === 'WORDS') return labWord(p);
    if (t.name === 'MEMORY') return p.map(function (b) { return [b]; });
    return labBoard(p);
  }
  function padExpected() { // what the answer should be, when it can be known: [outputs, text]
    var t = state.task, p = state.pad, o;
    if (t.name === 'MEMORY') return [[p[0] ^ p[4]], 'first ' + p[0] + ', last ' + p[4] + ': ' + (p[0] ^ p[4] ? 'they differ (1)' : 'the same (0)')];
    if (t.name === 'TICTACTOE') { var w = labWinner(p); if (w === 3) return [null, 'not a legal board: both win']; o = [w === 1 ? 1 : 0, w === 2 ? 1 : 0, w === 0 ? 1 : 0]; return [o, t.classes[best(o)]]; }
    if (t.name === 'DIGITS') {
      var near = 0, dist = 99;
      for (var d = 0; d < 10; d++) { var f = labDigit(d), k = 0; for (var i = 0; i < 35; i++) if (f[i] !== p[i]) k++; if (k < dist) { dist = k; near = d; } }
      o = []; for (var c = 0; c < 10; c++) o.push(c === near ? 1 : 0);
      return [o, dist === 0 ? 'the digit ' + near + ' of the font' : 'closest digit of the font: ' + near + ' (' + dist + ' px away)'];
    }
    var lang = ['en', 'fr'].filter(function (l) { return LAB_WORDS[l].split(' ').indexOf(p) >= 0; })[0];
    if (!lang) return [null, p ? 'a word the networks never saw' : 'type a word'];
    var at = t.all.label.indexOf(p);
    return [lang === 'en' ? [1, 0] : [0, 1], (lang === 'en' ? 'English' : 'French') + ' (' + (at >= 0 && t.isTest[at] ? 'a test word, never trained' : 'a trained word') + ')'];
  }
  function padChanged() { // the panel changed: its outputs (the server answers for its brain network)
    state.custom = padEncode();
    if (state.engine !== 'server' || state.remote === null) { draw(); return; }
    var t = state.task, pts = points(t), sample = shownSample(), id = state.build;
    LAB_SERVER.post('/api/snapshot', { id: state.remote, points: pts, sample: sample }).then(function (data) { if (id === state.build) { remoteUse(data, t, pts, sample); draw(); } }).catch(function (e) { engineFail(e, id); });
  }
  function padBuild() { // the panel of the task (kept when the same task is built again)
    var t = state.task, card = $('inputCard'), pad = $('inputPad'), kind = ['DIGITS', 'WORDS', 'MEMORY', 'TICTACTOE'].indexOf(t.name) >= 0 ? t.name : null;
    card.style.display = kind ? '' : 'none';
    if (!kind) { state.pad = null; state.custom = null; pad.removeAttribute('data-kind'); return; }
    if (pad.getAttribute('data-kind') !== kind) {
      pad.setAttribute('data-kind', kind);
      state.pad = kind === 'DIGITS' ? labDigit(3) : (kind === 'WORDS' ? 'soleil' : (kind === 'MEMORY' ? [1, 0, 1, 1, 0] : [1, 2, 0, 0, 1, 0, 2, 0, 1]));
      padRender();
    }
    state.custom = padEncode();
  }
  function padRender() {
    var t = state.task, pad = $('inputPad'), p = state.pad, kind = pad.getAttribute('data-kind'), s = '';
    if (kind === 'DIGITS') {
      s = '<div class="pad" style="grid-template-columns:repeat(5,30px)" aria-label="5 x 7 pixels: click or drag to draw">';
      for (var i = 0; i < 35; i++) s += '<button type="button" data-cell="' + i + '" aria-pressed="' + (p[i] ? 'true' : 'false') + '" aria-label="pixel ' + (i % 5 + 1) + ', ' + (Math.floor(i / 5) + 1) + '"></button>';
      s += '</div><div class="pad-tools"><select id="padDigit" aria-label="Start from a digit">' + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (d) { return '<option>' + d + '</option>'; }).join('') + '</select><button type="button" data-pad="digit">Load</button><button type="button" data-pad="noise">Flip 2 px</button><button type="button" data-pad="clear">Clear</button></div>';
    } else if (kind === 'WORDS') {
      s = '<div class="pad-tools"><label>Word (a–z, 6 letters) <input type="text" id="padWord" maxlength="6" spellcheck="false" autocomplete="off" value="' + p + '"></label></div><p class="hint">Accents are dropped (é = e); the letters after the 6th are not read.</p>';
    } else if (kind === 'MEMORY') {
      s = '<div class="pad" style="grid-template-columns:repeat(5,54px)" aria-label="The bits, one per time step">';
      for (var b = 0; b < 5; b++) s += '<button type="button" class="big" data-bit="' + b + '" aria-pressed="' + (p[b] ? 'true' : 'false') + '" style="width:54px;height:54px;font-size:22px">' + p[b] + '</button>';
      s += '</div><p class="hint">Step 1 to 5, left to right: the brain network reads one bit per time step.</p>';
    } else {
      s = '<div class="pad board" style="grid-template-columns:repeat(3,54px)" aria-label="Board: click a cell for X, O, empty">';
      for (var c = 0; c < 9; c++) s += '<button type="button" data-sq="' + c + '" aria-label="cell ' + (c + 1) + '">' + ['', 'X', 'O'][p[c]] + '</button>';
      s += '</div><div class="pad-tools"><button type="button" data-pad="empty">Empty board</button></div>';
    }
    pad.innerHTML = s;
    if (kind === 'WORDS') $('padWord').addEventListener('input', function () { state.pad = this.value.toLowerCase().normalize('NFD').replace(/[^a-z]/g, '').slice(0, 6); padChanged(); });
  }
  function drawInput() {
    var t = state.task;
    if (!state.pad) return;
    var exp = padExpected(), x = state.custom, a = state.layer.forward(x), c = state.brain.forward(x);
    var cell = function (v) {
      if (!t.classes) return '<td class="' + (exp[0] ? (labGood(t, v, exp[0]) ? 'ok' : 'bad') : '') + '">' + f2(v[0]) + '</td>';
      var top = best(v);
      return '<td class="' + (exp[0] ? (labGood(t, v, exp[0]) ? 'ok' : 'bad') : '') + '"><div class="bars">' + v.map(function (y, k) { return '<div class="bar' + (k === top ? ' top' : '') + '"><span>' + t.classes[k] + '</span><i style="width:' + Math.round(100 * Math.max(0, Math.min(1, y))) + 'px"></i>' + f2(y) + '</div>'; }).join('') + '</div></td>';
    };
    $('inputOut').innerHTML = '<thead><tr><th>Expected</th><th>Layered</th><th>Brain</th></tr></thead><tbody><tr><td>' + exp[1] + '</td>' + cell(a) + cell(c) + '</tr></tbody>';
    $('in-info').textContent = { DIGITS: 'draw a digit (click or drag on the pixels)', WORDS: 'type a word: English or French?', MEMORY: 'the bits of the sequence: click to flip one', TICTACTOE: 'click a cell: empty, X, O' }[t.name];
  }
  var painting = null; // the value a drag on the digit paints
  $('inputPad').addEventListener('pointerdown', function (e) {
    var b = e.target.closest('button'); if (!b || !state.pad) return;
    var t = state.task, p = state.pad;
    if (b.hasAttribute('data-cell')) { var i = +b.getAttribute('data-cell'); painting = 1 - p[i]; p[i] = painting; b.setAttribute('aria-pressed', painting ? 'true' : 'false'); b.releasePointerCapture && b.releasePointerCapture(e.pointerId); padChanged(); e.preventDefault(); return; }
    if (b.hasAttribute('data-bit')) { var k = +b.getAttribute('data-bit'); p[k] = 1 - p[k]; padRender(); padChanged(); return; }
    if (b.hasAttribute('data-sq')) { var q = +b.getAttribute('data-sq'); p[q] = (p[q] + 1) % 3; padRender(); padChanged(); return; }
    var act = b.getAttribute('data-pad');
    if (act === 'digit') state.pad = labDigit(+$('padDigit').value);
    else if (act === 'clear') state.pad = new Array(35).fill(0);
    else if (act === 'noise') for (var n = 0; n < 2; n++) { var at = Math.floor(Math.random() * 35); p[at] = 1 - p[at]; }
    else if (act === 'empty') state.pad = new Array(9).fill(0);
    else return;
    var keep = t.name === 'DIGITS' ? $('padDigit').value : null; padRender(); if (keep !== null) $('padDigit').value = keep; padChanged();
  });
  $('inputPad').addEventListener('pointerover', function (e) { // drag: paint the pixels crossed
    if (painting === null || !(e.buttons & 1)) { painting = null; return; }
    var b = e.target.closest('[data-cell]'); if (!b) return;
    var i = +b.getAttribute('data-cell'); if (state.pad[i] === painting) return;
    state.pad[i] = painting; b.setAttribute('aria-pressed', painting ? 'true' : 'false'); padChanged();
  });
  document.addEventListener('pointerup', function () { painting = null; });
  $('inShow').addEventListener('change', function () { if (state.engine === 'server' && state.remote !== null) padChanged(); else draw(); });

  // ---------- events ----------
  $('task').innerHTML = LAB_TASKS.map(function (g) { return '<optgroup label="' + g.group + '">' + g.names.map(function (n) { return '<option' + (n === 'XOR' ? ' selected' : '') + '>' + n + '</option>'; }).join('') + '</optgroup>'; }).join('');
  $('bar').addEventListener('click', function (e) {
    var b = e.target.closest('[data-act]'); if (!b) return;
    var act = b.getAttribute('data-act');
    if (act === 'play') setPlay(!state.running);
    else if (act === 'round') { if (state.engine === 'server') serverRounds(1).then(draw); else { trainRound(); draw(); } }
    else if (act === 'ten') { if (state.engine === 'server') serverRounds(100).then(draw); else { for (var r = 0; r < 100; r++) trainRound(); draw(); } }
    else if (act === 'reset') build(); // Play keeps running on the new network
    else if (act === 'regen') regenerate();
    else if (act === 'defaults') { restoreDefaults(true); build(); }
  });
  $('step').addEventListener('input', draw);
  $('sample').addEventListener('input', function () {
    if (state.engine !== 'server' || state.remote === null) { draw(); return; }
    var t = state.task, pts = points(t), sample = shownSample();
    var id = state.build;
    LAB_SERVER.post('/api/snapshot', { id: state.remote, points: pts, sample: sample }).then(function (data) { if (id === state.build) { remoteUse(data, t, pts, sample); draw(); } }).catch(function (e) { engineFail(e, id); });
  });
  // The engine: the server is checked first, the page engine stays when it does not answer
  function setEngine(value) {
    if (state.retry) { clearTimeout(state.retry); state.retry = null; }
    if (value !== 'server') { state.engine = 'js'; state.remote = null; state.creating = false; $('msg').textContent = ''; build(); return; }
    state.engine = 'server'; $('engine').value = 'server';
    $('msg').textContent = 'Connecting to the server engine…';
    LAB_SERVER.info(1500).then(function (info) { $('msg').textContent = 'Server engine' + (info.gpu ? ' (gpu: ' + info.gpu + ')' : ''); build(); }).catch(function (e) { e.network = true; engineLost(e); });
  }
  $('engine').addEventListener('change', function () { setEngine($('engine').value); });
  $('engineGpu').addEventListener('change', function () { if (state.engine === 'server') build(); });
  // The values written in the page are the defaults: the browser must not restore the ones of an older visit
  var fields = document.querySelectorAll('.lab input, .lab select'), defaults = new Map();
  Array.prototype.forEach.call(fields, function (el) {
    el.setAttribute('autocomplete', 'off');
    if (el.id === 'sample' || el.id === 'engine' || el.id === 'engineGpu' || el.closest('#l-list')) return; // filled by the page (the samples, the hidden layers) or the choice of the engine
    var selected = el.tagName === 'SELECT' ? (Array.prototype.find.call(el.options, function (o) { return o.defaultSelected; }) || el.options[0]) : null;
    defaults.set(el, el.type === 'checkbox' ? el.defaultChecked : (el.tagName === 'SELECT' ? (selected ? selected.value : '') : el.defaultValue));
  });
  // keepTask: the task stays, with its preset (a scenario), else everything back (XOR)
  function restoreDefaults(keepTask) {
    var task = $('task').value;
    defaults.forEach(function (value, el) { if (el.type === 'checkbox') el.checked = value; else el.value = value; });
    $('l-list').innerHTML = ''; layerField(4); // one hidden layer of 4 neurons
    $('batchWorkers').value = Math.max(1, Math.min(16, (navigator.hardwareConcurrency || 4)));
    if (keepTask) { $('task').value = task; applyPreset(); }
  }
  // The settings of a scenario (LAB_PRESETS, measured on the server engine): set on the defaults when it is chosen
  function applyPreset() {
    var p = LAB_PRESETS[$('task').value];
    state.preset = !!p; $('presetInfo').textContent = p ? p.note : '';
    if (!p) return;
    Object.keys(p.fields).forEach(function (id) { var el = $(id); if (el.type === 'checkbox') el.checked = p.fields[id]; else el.value = p.fields[id]; });
    $('l-list').innerHTML = ''; p.layers.forEach(layerField);
  }
  restoreDefaults(); // also undoes what the browser restored by itself
  $('l-add').addEventListener('click', function () { var last = hidden(); layerField(last.length ? last[last.length - 1] : 4); build(); });
  $('l-del').addEventListener('click', function () { var list = $('l-list'); if (list.lastElementChild) list.removeChild(list.lastElementChild); build(); }); // no hidden layer: the inputs go straight to the output
  ['density', 'density2d', 'noise', 'testPct', 'dataSeed'].forEach(function (id) { $(id).addEventListener('change', build); });
  $('task').addEventListener('change', function () { // a scenario comes with its settings; leaving one gives the defaults back
    if (LAB_PRESETS[$('task').value] || state.preset) restoreDefaults(true);
    build();
  });
  Array.prototype.forEach.call(document.querySelectorAll('.lab-form input, .lab-form select'), function (el) { el.addEventListener('change', build); }); // a parameter rebuilds the network with its seed, Play keeps running

  // ---------- batch: N networks of each kind with the current parameters, spread over Web Workers ----------
  var batch = { workers: [], results: { layer: [], brain: [] }, start: 0, total: 0 };
  function batchJobs() {
    var n = Math.max(1, Math.round(num('batchN', 100))), rounds = Math.max(1, Math.round(num('batchRounds', 500))), common = { task: $('task').value, taskOpt: taskOpt(), rounds: rounds };
    var seeds = function () { var s = []; for (var i = 0; i < n; i++) s.push(1 + Math.floor(Math.random() * 2147483646)); return s; };
    return [
      Object.assign({ kind: 'layer', seeds: seeds(), hidden: hidden(), rate: num('l-rate', 1.5), coef: $('l-coef').value, layerOpt: { activation: $('l-act').value, range: num('l-range', 1), weightInit: $('l-wi').checked, biasInit: $('l-bi').checked } }, common),
      Object.assign({ kind: 'brain', seeds: seeds(), brainOpt: brainOpt(1), train: brainTrain(), filter: Math.round(num('b-filter', 0)), allInputs: $('b-allin').checked }, common)
    ];
  }
  function median(v) { var s = v.slice().sort(function (a, b) { return a - b; }); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN; }
  function mean(v) { return v.length ? v.reduce(function (a, b) { return a + b; }, 0) / v.length : NaN; }
  function batchDraw() {
    var withTest = state.task.test.in.length > 0;
    var rows = '<thead><tr><th>Kind</th><th>Done</th><th>Solved</th><th>Mean error</th><th>Median error</th><th>Best / worst</th>' + (withTest ? '<th>Test error (mean)</th><th>Test solved</th>' : '') + '<th>Rounds to solve (mean)</th><th>Train time / network</th><th>Train time (sum)</th></tr></thead><tbody>';
    ['layer', 'brain'].forEach(function (k) {
      var r = batch.results[k], errs = r.map(function (x) { return x.error; }).filter(isFinite), solved = r.filter(function (x) { return x.solved; }), times = r.map(function (x) { return x.ms; });
      var at = r.filter(function (x) { return x.solvedAt > 0; }).map(function (x) { return x.solvedAt; }), sum = times.reduce(function (a, b) { return a + b; }, 0), terr = r.map(function (x) { return x.testError; }).filter(isFinite);
      rows += '<tr><td>' + (k === 'layer' ? 'Layered' : 'Brain') + '</td><td>' + r.length + ' / ' + batch.total + '</td><td>' + (r.length ? (100 * solved.length / r.length).toFixed(0) + ' %' : '–') + '</td><td>' + (errs.length ? f2(mean(errs)) : '–') + '</td><td>' + (errs.length ? f2(median(errs)) : '–') + '</td><td>' + (errs.length ? f2(Math.min.apply(null, errs)) + ' / ' + f2(Math.max.apply(null, errs)) : '–') + '</td>'
        + (withTest ? '<td>' + (terr.length ? f2(mean(terr)) : '–') + '</td><td>' + (r.length ? (100 * r.filter(function (x) { return x.testSolved; }).length / r.length).toFixed(0) + ' %' : '–') + '</td>' : '')
        + '<td>' + (at.length ? mean(at).toFixed(0) : '–') + '</td><td>' + (r.length ? ms(sum / r.length) : '–') + '</td><td>' + ms(sum) + '</td></tr>';
    });
    $('batchSummary').innerHTML = rows + '</tbody>';
    // histogram of the final errors (10 bins from 0 to 0.5 and more)
    var W = 760, H = 200, l = 40, b = 28, bins = 10, s = '';
    var count = function (r) { var c = new Array(bins).fill(0); r.forEach(function (x) { c[Math.min(bins - 1, Math.floor((isFinite(x.error) ? x.error : 1) / 0.5 * bins))]++; }); return c; };
    var cl = count(batch.results.layer), cb = count(batch.results.brain), max = Math.max.apply(null, cl.concat(cb, [1])), bw = (W - l - 10) / bins;
    for (var i = 0; i < bins; i++) {
      var x = l + i * bw, hl = (H - b - 10) * cl[i] / max, hb = (H - b - 10) * cb[i] / max;
      s += '<rect x="' + (x + 2) + '" y="' + (H - b - hl) + '" width="' + (bw / 2 - 3) + '" height="' + hl + '" style="fill:var(--accent)"/><rect x="' + (x + bw / 2 + 1) + '" y="' + (H - b - hb) + '" width="' + (bw / 2 - 3) + '" height="' + hb + '" style="fill:var(--warn)"/>';
      s += '<text x="' + (x + bw / 2) + '" y="' + (H - 10) + '" text-anchor="middle">' + (i === bins - 1 ? '≥ 0.45' : (i * 0.05).toFixed(2)) + '</text>';
    }
    s += '<line class="axis" x1="' + l + '" x2="' + (W - 10) + '" y1="' + (H - b) + '" y2="' + (H - b) + '"/><text x="' + l + '" y="12">networks per final error (blue layered, orange brain), max ' + max + '</text>';
    $('batchHist').innerHTML = s;
    var list = '<thead><tr><th>#</th><th>Layered seed</th><th>Error</th>' + (withTest ? '<th>Test</th>' : '') + '<th>Solved</th><th>Time</th><th>Brain seed</th><th>Error</th>' + (withTest ? '<th>Test</th>' : '') + '<th>Solved</th><th>Hops</th><th>Time</th></tr></thead><tbody>';
    for (var j = 0; j < batch.total; j++) {
      var a = batch.results.layer.find(function (x) { return x.index === j; }), c = batch.results.brain.find(function (x) { return x.index === j; });
      if (!a && !c) continue;
      list += '<tr><td>' + (j + 1) + '</td>' + (a ? '<td>' + a.seed + '</td><td>' + f2(a.error) + '</td>' + (withTest ? '<td>' + f2(a.testError) + '</td>' : '') + '<td class="' + (a.solved ? 'ok' : 'bad') + '">' + (a.solved ? 'yes' : 'no') + '</td><td>' + ms(a.ms) + '</td>' : '<td colspan="' + (withTest ? 5 : 4) + '">…</td>')
        + (c ? '<td>' + c.seed + '</td><td>' + f2(c.error) + '</td>' + (withTest ? '<td>' + f2(c.testError) + '</td>' : '') + '<td class="' + (c.solved ? 'ok' : 'bad') + '">' + (c.solved ? 'yes' : 'no') + '</td><td>' + (c.hops < 0 ? 'none' : c.hops) + '</td><td>' + ms(c.ms) + '</td>' : '<td colspan="' + (withTest ? 6 : 5) + '">…</td>') + '</tr>';
    }
    $('batchRows').innerHTML = list + '</tbody>';
    var left = batch.total * 2 - batch.results.layer.length - batch.results.brain.length;
    $('batchState').textContent = left ? 'Running: ' + left + ' network(s) left, ' + batch.workers.length + ' worker(s), ' + ms(performance.now() - batch.start) + ' so far.' : 'Done: ' + batch.total * 2 + ' networks in ' + ms(performance.now() - batch.start) + ' (wall time, ' + batch.workers.length + ' worker(s) in parallel).';
  }
  function batchStop() { batch.workers.forEach(function (w) { if (w.terminate) w.terminate(); }); batch.workers = []; $('batchRun').disabled = false; $('batchStop').disabled = true; }
  function batchStart() {
    batchStop();
    var jobs = batchJobs(), wanted = Math.max(1, Math.round(num('batchWorkers', 4))), src = window.LAB_WORKER_SRC, last = 0;
    batch.results = { layer: [], brain: [] }; batch.total = jobs[0].seeds.length; batch.start = performance.now();
    $('batchRun').disabled = true; $('batchStop').disabled = false;
    var parts = [], perKind = Math.max(1, Math.round(wanted / jobs.length)); // the workers asked, shared by the two kinds
    if (state.engine === 'server') { // the brain networks on the server, a few seeds per request (the progress shows), the layered ones in the workers
      var brainJob = jobs[1], sopt = serverOptions(1), stask = serverTask(makeTask(brainJob.task, brainJob.taskOpt)), at = 0, chunk = 4;
      var stop = { terminate: function () { at = brainJob.seeds.length; } };
      batch.workers.push(stop);
      (function next() {
        if (at >= brainJob.seeds.length) return;
        var seeds = brainJob.seeds.slice(at, at + chunk), first = at; at += chunk;
        LAB_SERVER.post('/api/batch', { options: sopt, task: stask, rounds: brainJob.rounds, train: Object.assign(brainTrain(), { revive: $('b-revive').checked }), seeds: seeds }).then(function (list) {
          list.forEach(function (res, k) { res.index = first + k; batch.results.brain.push(res); });
          batchDraw();
          if (batch.results.layer.length + batch.results.brain.length >= batch.total * 2) batchStop(); else next();
        }).catch(function (e) { batchStop(); $('batchState').textContent = 'Server batch failed: ' + e.message; });
      })();
      jobs = [jobs[0]]; perKind = wanted;
    }
    jobs.forEach(function (job) { var per = Math.ceil(job.seeds.length / perKind); for (var w = 0; w < perKind; w++) { var seeds = job.seeds.slice(w * per, (w + 1) * per); if (seeds.length) parts.push(Object.assign({}, job, { seeds: seeds, offset: w * per })); } });
    var url = null;
    try { url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' })); } catch (e) { url = null; }
    parts.forEach(function (part) {
      var onResult = function (msg) {
        if (msg.done) return; // the last result already ended the batch
        msg.index += part.offset; batch.results[part.kind].push(msg);
        if (performance.now() - last > 150) { last = performance.now(); batchDraw(); }
        if (batch.results.layer.length + batch.results.brain.length >= batch.total * 2) { batchDraw(); batchStop(); }
      };
      var worker = null;
      try { worker = url ? new Worker(url) : null; } catch (e) { worker = null; }
      if (worker) { worker.onmessage = function (e) { onResult(e.data); }; worker.postMessage(part); batch.workers.push(worker); }
      else { // no worker (blocked): on the page, one network at a time
        var i = 0, fake = { terminate: function () { i = part.seeds.length; } };
        batch.workers.push(fake);
        (function next() { if (i >= part.seeds.length) return; onResult(batchRun(part, i++)); setTimeout(next, 0); })();
      }
    });
    batchDraw();
  }
  $('batchRun').addEventListener('click', batchStart);
  if (LAB_SERVER.base === '') { build(); setEngine('server'); return; } // the page served by lab/server.sh: its engine first
  $('batchStop').addEventListener('click', function () { batchStop(); batchDraw(); $('batchState').textContent += ' (cancelled)'; });
  build();
});
