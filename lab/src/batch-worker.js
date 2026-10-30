// Batch worker: trains networks of one kind on a task, one seed after the other, and sends each result back (runs inside a Web Worker, or on the page without one)
function batchRun(job, index) {
  var clock = typeof performance !== 'undefined' ? performance : Date, start = clock.now(), task = makeTask(job.task, job.taskOpt), net, seed = job.seeds[index], hops = null, solvedAt = -1;
  if (job.kind === 'layer') net = new LayerSim([task.layerInputs].concat(job.hidden, [task.outputs]), seed, job.layerOpt);
  else {
    for (var tries = 0; tries < 500; tries++) { // the same filter as Regenerate: the outputs at most `filter` hops from an input
      net = new BrainSim(Object.assign({}, job.brainOpt, { seed: seed })); hops = job.allInputs ? net.hopsAll() : net.hops();
      if (!job.filter || (hops >= 1 && hops <= job.filter)) break;
      seed = (seed * 48271 + 11) % 2147483647 || 1;
    }
  }
  for (var r = 0; r < job.rounds; r++) {
    if (job.kind === 'layer') for (var i = 0; i < task.in.length; i++) net.learn(labLayerCoef(job.coef, job.rate, net.forward(task.in[i]), task.out[i]));
    else labBrainRound(net, task, job.train, r);
    if (solvedAt < 0 && labSolved(task, net, task.in, task.clean)) solvedAt = r + 1;
  }
  var error = labError(net, task.in, task.clean);
  return { index: index, seed: seed, error: isFinite(error) ? error : NaN, solved: labSolved(task, net, task.in, task.clean), solvedAt: solvedAt,
    testError: labError(net, task.test.in, task.test.out), testSolved: labSolved(task, net, task.test.in, task.test.out), hops: job.kind === 'brain' ? (job.allInputs ? net.hopsAll() : net.hops()) : null, ms: clock.now() - start };
}
if (typeof self !== 'undefined' && typeof window === 'undefined') {
  self.onmessage = function (e) { var job = e.data; for (var i = 0; i < job.seeds.length; i++) self.postMessage(batchRun(job, i)); self.postMessage({ done: true }); };
}
