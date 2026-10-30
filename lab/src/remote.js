// Server engine of the lab: the brain network runs with the C++ library (lab/server.sh), the page draws a mirror of its last snapshot
var LAB_SERVER = (function () {
  var base = /^https?:$/.test(location.protocol) ? '' : 'http://localhost:8765'; // same origin when the server serves the page

  // A POST to the server: the JSON body ends with an empty line (the end of a payload for the server), text/plain avoids the CORS preflight
  function post(path, body) {
    return fetch(base + path, { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: JSON.stringify(body) + '\r\n\r\n' }).catch(function (e) {
      var lost = new Error(e && e.message ? e.message : 'no answer'); lost.network = true; throw lost; // the server is down (not an answer of it)
    }).then(function (r) {
      return r.json().then(function (data) { if (!r.ok || data.error) throw new Error(data.error || ('HTTP ' + r.status)); return data; });
    });
  }
  // The engine answers within the time given (else it is down)
  function info(timeout) {
    var done = false;
    return new Promise(function (resolve, reject) {
      setTimeout(function () { if (!done) { done = true; reject(new Error('no answer from the server')); } }, timeout || 1500);
      fetch(base + '/api/info').then(function (r) { return r.json(); }).then(function (data) { if (!done) { done = true; resolve(data); } }).catch(function (e) { if (!done) { done = true; reject(e); } });
    });
  }
  return { post: post, info: info, base: base };
})();

// The mirror of a network of the server: what the drawing and the measures read (forward only answers the points sent)
function RemoteBrain(snapshot, points, opt, sample) {
  var n = snapshot.n, cap = snapshot.cap, self = this, used = new Array(n).fill(0);
  this.remote = true; this.n = n; this.cap = cap; this.steps = snapshot.steps; this.input = snapshot.input; this.output = snapshot.output; this.opt = opt;
  this.pos = snapshot.pos.map(function (p) { return { x: p[0], y: p[1] }; }); this.bias = snapshot.bias; this.snapshot = snapshot;
  this.links = []; for (var i = 0; i < n * cap; i++) this.links.push(null);
  snapshot.links.forEach(function (l) { self.links[l[0] * cap + used[l[0]]++] = { id: l[1], w: l[2], inward: true }; }); // <receiver, sender, weight>, inward slots packed
  this.outputs = {}; (points || []).forEach(function (p, k) { self.outputs[p.join(',')] = snapshot.outputs[k]; });
  this.sampleKey = sample ? sample.join(',') : null; this.states = snapshot.states || [];
}
RemoteBrain.prototype.forward = function (x) { var key = x.join(','), out = this.outputs[key]; if (key === this.sampleKey && this.snapshot.states) this.states = this.snapshot.states; return out ? out.slice() : new Array(this.output || 1).fill(NaN); };
RemoteBrain.prototype.hops = function () { return this.snapshot.hops; };
RemoteBrain.prototype.hopsAll = function () { return this.snapshot.hopsAll; };
