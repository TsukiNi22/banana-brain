// Tasks of the lab: makeTask(name, {density (curves), density2d (2D grids and spirals), noise, test, seed}) -> {in, out (one array per sample), test: {in, out}, kind, dims, outputs}
// kind: 'class' (every output on the right side of 0.5) | 'reg' (every output within 0.1); dims: 1 or 2 for the continuous tasks (drawn as a curve / a map)
var LAB_TASKS = [
  { group: 'Logic', names: ['AND', 'OR', 'NAND', 'XOR', 'COPY', 'NOT', 'MAJ3', 'PAR3', 'MUX3', 'CMP2', 'MAJ5', 'PAR4', 'PAR5', 'MAJ7', 'PAR6', 'CMP3'] },
  { group: 'Several outputs', names: ['ADD2', 'DEC2', 'ADD3', 'MUL2', 'SEG7'] },
  { group: 'Curves (x -> y)', names: ['SIN', 'SIN4', 'X2', 'BELL', 'STEP', 'ABS', 'SAW', 'SINC', 'SMOOTH', 'CHIRP', 'STAIRS'] },
  { group: '2D shapes (x, y)', names: ['CIRCLE', 'CHECKER', 'SPIRALS', 'RING', 'MOONS', 'TRIANGLE', 'DIAMOND', 'CROSS', 'STRIPES', 'QUADS'] },
  { group: '2D functions (x, y)', names: ['SINCOS', 'BUMP2D', 'RIPPLE', 'SADDLE'] },
  { group: '3D (x, y, z)', names: ['SPHERE', 'SHELL', 'OCTANTS', 'CHECK3', 'PLANE3D', 'GAUSS3D', 'SINE3D'] },
  { group: 'Scenarios (presets, your own inputs)', names: ['DIGITS', 'WORDS', 'MEMORY', 'TICTACTOE'] }
];
var LAB_TASK_INFO = {
  AND: 'a AND b', OR: 'a OR b', NAND: 'NOT (a AND b)', XOR: 'a XOR b (not linearly separable)', COPY: 'output = a', NOT: 'output = NOT a',
  MAJ3: 'majority of 3 inputs', PAR3: 'parity of 3 inputs', MUX3: 's ? b : a (the first input selects one of the two others)', CMP2: 'a > b, a and b on 2 bits (4 inputs)',
  MAJ5: 'majority of 5 inputs (32 samples)', PAR4: 'parity of 4 inputs (16 samples)', PAR5: 'parity of 5 inputs (32 samples, the hardest logic task)',
  ADD2: 'a + b, 2 bits each: 4 inputs, 3 outputs (the sum in binary)', DEC2: 'decoder 2 -> 4: one output at 1 among 4',
  SIN: '(sin(2 pi x) + 1) / 2, one period', SIN4: '(sin(4 pi x) + 1) / 2, two periods', X2: 'x squared', BELL: 'gaussian bump exp(-((x - 0.5) / 0.15)^2)', STEP: '0 under x = 0.5, 1 from it',
  DIGITS: 'the digit drawn on a 5 x 7 grid: 35 inputs, 10 outputs (noisy variants of each digit)', WORDS: 'the language of a word (English / French), 5 bits per letter, up to 6 letters: 30 inputs, 2 outputs',
  MEMORY: 'a bit arrives at each time step (5 steps): output 1 when the first and the last bits differ (the brain must remember; the layered network sees them all at once)',
  TICTACTOE: 'who wins on a 3 x 3 board (X plane, O plane: 18 inputs): X wins, O wins, nobody',
  MAJ7: 'majority of 7 inputs (128 samples)', PAR6: 'parity of 6 inputs (64 samples)', CMP3: 'a > b, a and b on 3 bits (6 inputs, 64 samples)',
  ADD3: 'full adder: a + b + carry in, 3 inputs, 2 outputs (the sum bit and the carry out)', MUL2: 'a x b, 2 bits each: 4 inputs, 4 outputs (the product in binary)', SEG7: 'a digit 0 - 9 on 4 bits -> the 7 segments of a display (7 outputs)',
  ABS: '|2x - 1| (a V)', SAW: 'sawtooth, 3 teeth', SINC: 'sinc: sin(10 (x - 0.5)) / (10 (x - 0.5)), a peak with ripples', SMOOTH: 'smoothstep: 3x^2 - 2x^3', CHIRP: 'sin of a growing frequency (hard to learn on the right side)', STAIRS: 'a staircase of 4 steps',
  RING: '1 between two circles (radius 0.18 and 0.36)', MOONS: 'two interleaved half-circles (the classic moons)', TRIANGLE: '1 inside a triangle', DIAMOND: '1 inside a diamond |x| + |y| < 0.3', CROSS: '1 on a plus sign', STRIPES: 'diagonal stripes', QUADS: 'XOR of the two halves: opposite quadrants',
  BUMP2D: 'a 2D gaussian bump (regression)', RIPPLE: 'rings around the center fading out (regression)', SADDLE: '(x - 0.5)(y - 0.5): a saddle (regression)',
  SPHERE: '1 inside a ball of radius 0.35 at the center of the cube (3 inputs, drawn in 3 slices)', SHELL: '1 between two spheres (radius 0.25 and 0.4)', OCTANTS: 'XOR of the three halves: parity of the octant', CHECK3: '3D checkerboard of 3 x 3 x 3 cells',
  PLANE3D: '1 above the plane x + y + z = 1.5 (linearly separable)', GAUSS3D: 'a 3D gaussian bump (regression)', SINE3D: 'sin(2 pi x) cos(pi y) z (regression)',
  CIRCLE: '1 inside a disc of radius 0.3 at the center', CHECKER: 'checkerboard of 3 x 3 cells', SPIRALS: 'two interleaved spirals (the classic hard one)', SINCOS: '(sin(pi x) cos(pi y) + 1) / 2 (regression)'
};
// The 5 x 7 font of the digits (rows, '#' = on) and the words of the two languages (a - z, 6 letters at most)
var LAB_DIGITS = [
  [' ### ', '#   #', '#  ##', '# # #', '##  #', '#   #', ' ### '], ['  #  ', ' ##  ', '  #  ', '  #  ', '  #  ', '  #  ', ' ### '],
  [' ### ', '#   #', '    #', '   # ', '  #  ', ' #   ', '#####'], ['#####', '   # ', '  #  ', '   # ', '    #', '#   #', ' ### '],
  ['   # ', '  ## ', ' # # ', '#  # ', '#####', '   # ', '   # '], ['#####', '#    ', '#### ', '    #', '    #', '#   #', ' ### '],
  ['  ## ', ' #   ', '#    ', '#### ', '#   #', '#   #', ' ### '], ['#####', '    #', '   # ', '  #  ', ' #   ', ' #   ', ' #   '],
  [' ### ', '#   #', '#   #', ' ### ', '#   #', '#   #', ' ### '], [' ### ', '#   #', '#   #', ' ####', '    #', '   # ', ' ##  ']
];
var LAB_WORDS = {
  en: 'the and with water house night light green white black happy money small world thing think where which would could about after again know want work year good time day way man new first last long great little right old big high young child woman sound earth paper story party power heart music point table watch fight dream sleep',
  fr: 'maison chat chien noir blanc rouge vert bleu jour nuit petit grand beau belle avec dans pour sans sous chez quand comme mais donc alors aussi jamais encore bien tres trois quatre cinq deux homme femme enfant pain feu monde pays ville route porte fleur arbre soleil plage lune etoile voiture ecole livre'
};
// The settings of the scenarios (form ids -> values, the hidden layers of the layered network), set when one is chosen; note: what was measured
var LAB_PRESETS = { // measured once per setting (one seed, the server engine on the host, a loaded machine): the time is an order of magnitude; the links at the init and the minimum of links are the ones of the measures (the defaults of the library are a random count and 1)
  DIGITS: { fields: { testPct: 0.25, 'b-linit': 'max', 'b-minlinks': 0, 'b-neurons': 80, 'b-shook': 8, 'b-filter': 0, 'l-rate': 1.5 }, layers: [20],
    note: 'Preset: 80 hidden neurons, 8 time steps, no path filter (35 inputs); layered {20}. Measured: the brain is solved in about 700 rounds, 15 s on the server engine (about 55 ms a round on the page engine), 83 % of the test digits right; the layered one in 20 rounds. Draw a digit, flip pixels: which one does each network see?' },
  WORDS: { fields: { testPct: 0.25, 'b-linit': 'max', 'b-minlinks': 0, 'b-neurons': 80, 'b-shook': 8, 'b-filter': 0, 'l-rate': 0.5 }, layers: [64, 16],
    note: 'Preset: 80 hidden neurons, 8 time steps, no path filter (30 inputs); layered {64, 16}. Measured: the brain is solved in about 90 rounds, 2 s on the server engine (about 50 ms a round on the page engine); about 60 % (brain) and 70 % (layered) of the test words right: the letters only give hints. Type a word of your own.' },
  MEMORY: { fields: { testPct: 0.25, 'b-linit': 'max', 'b-minlinks': 0, 'b-neurons': 40, 'b-shook': 8, 'l-rate': 1.5 }, layers: [8],
    note: 'Preset: 40 hidden neurons, 8 time steps (the 5 bits, then 3 steps to answer: with the default 20 steps the first bit is forgotten and nothing is learned). Measured: solved in 650 to 900 rounds, about 2 s on the server engine (4 seeds), every test sequence right; the layered network sees the 5 bits at once (no memory needed). Flip the bits: only the first and the last count.' },
  TICTACTOE: { fields: { testPct: 0.25, 'b-linit': 'max', 'b-minlinks': 0, 'b-neurons': 80, 'b-shook': 8, 'b-filter': 0, 'l-rate': 1.5 }, layers: [36],
    note: 'Preset: 80 hidden neurons, 8 time steps, no path filter (18 inputs), 600 boards (200 of each result); layered {36}. Measured: the brain is solved in about 190 rounds, 40 s on the server engine (the page engine is several times slower: use the server), 86 % of the test boards right, like the layered one (4 s). Set a board and see who wins.' }
};
function labDigit(d) { var v = []; LAB_DIGITS[d].forEach(function (row) { for (var c = 0; c < 5; c++) v.push(row[c] === '#' ? 1 : 0); }); return v; } // 35 pixels, row after row
function labWord(word) { var v = []; for (var p = 0; p < 6; p++) { var c = p < word.length ? word.charCodeAt(p) - 96 : 0; if (c < 0 || c > 26) c = 0; for (var b = 4; b >= 0; b--) v.push((c >> b) & 1); } return v; } // 6 letters x 5 bits (a = 1 ... z = 26, none = 0)
function labWinner(board) { // board: 9 cells 0 / 1 (X) / 2 (O) -> 1, 2 or 0
  var lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]], won = 0;
  lines.forEach(function (l) { if (board[l[0]] && board[l[0]] === board[l[1]] && board[l[1]] === board[l[2]]) won |= board[l[0]]; });
  return won; // 3 = both (not a legal board)
}
function labBoard(board) { var v = []; for (var c = 0; c < 9; c++) v.push(board[c] === 1 ? 1 : 0); for (var o = 0; o < 9; o++) v.push(board[o] === 2 ? 1 : 0); return v; } // X plane then O plane
function labBits(n) { var all = []; for (var i = 0; i < (1 << n); i++) { var v = []; for (var b = 0; b < n; b++) v.push((i >> b) & 1); all.push(v); } return all; }
var LAB_CURVES = { SIN: function (x) { return (Math.sin(2 * Math.PI * x) + 1) / 2; }, SIN4: function (x) { return (Math.sin(4 * Math.PI * x) + 1) / 2; }, X2: function (x) { return x * x; },
  BELL: function (x) { return Math.exp(-Math.pow((x - 0.5) / 0.15, 2)); }, STEP: function (x) { return x >= 0.5 ? 1 : 0; },
  ABS: function (x) { return Math.abs(2 * x - 1); }, SAW: function (x) { return (x * 3) % 1; }, SINC: function (x) { var u = 10 * (x - 0.5); return Math.abs(u) < 1e-9 ? 1 : (Math.sin(u) / u + 0.22) / 1.22; },
  SMOOTH: function (x) { return 3 * x * x - 2 * x * x * x; }, CHIRP: function (x) { return (Math.sin(2 * Math.PI * (x + 3 * x * x)) + 1) / 2; }, STAIRS: function (x) { return Math.min(3, Math.floor(x * 4)) / 3; } };
var LAB_CLASS_CURVES = ['STEP', 'STAIRS']; // the other curves are regressions
function labArcDistance(x, y, cx, cy, r, upper) { var px = x - cx, py = y - cy; if (upper ? py >= 0 : py <= 0) return Math.abs(Math.hypot(px, py) - r); return Math.min(Math.hypot(px - r, py), Math.hypot(px + r, py)); } // to a half circle
var LAB_SHAPES = {
  CIRCLE: function (x, y) { return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) < 0.09 ? 1 : 0; },
  CHECKER: function (x, y) { return (Math.min(2, Math.floor(x * 3)) + Math.min(2, Math.floor(y * 3))) % 2; },
  RING: function (x, y) { var r = Math.hypot(x - 0.5, y - 0.5); return r > 0.18 && r < 0.36 ? 1 : 0; },
  MOONS: function (x, y) { return labArcDistance(x, y, 0.38, 0.55, 0.25, true) < labArcDistance(x, y, 0.62, 0.45, 0.25, false) ? 1 : 0; },
  TRIANGLE: function (x, y) { return y > 0.15 && y < 1.7 * (x - 0.15) + 0.15 && y < -1.7 * (x - 0.85) + 0.15 ? 1 : 0; },
  DIAMOND: function (x, y) { return Math.abs(x - 0.5) + Math.abs(y - 0.5) < 0.3 ? 1 : 0; },
  CROSS: function (x, y) { var dx = Math.abs(x - 0.5), dy = Math.abs(y - 0.5); return (dx < 0.1 || dy < 0.1) && Math.max(dx, dy) < 0.38 ? 1 : 0; },
  STRIPES: function (x, y) { return Math.floor((x + y) * 3) % 2; },
  QUADS: function (x, y) { return (x > 0.5) !== (y > 0.5) ? 1 : 0; },
  SINCOS: function (x, y) { return (Math.sin(Math.PI * x) * Math.cos(Math.PI * y) + 1) / 2; },
  BUMP2D: function (x, y) { return Math.exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.05); },
  RIPPLE: function (x, y) { var r = Math.hypot(x - 0.5, y - 0.5); return (Math.sin(14 * r) * Math.exp(-3 * r) * 1.6 + 1) / 2; },
  SADDLE: function (x, y) { return (x - 0.5) * (y - 0.5) * 2 + 0.5; } };
var LAB_REG2 = ['SINCOS', 'BUMP2D', 'RIPPLE', 'SADDLE'];
var LAB_SOLIDS = {
  SPHERE: function (x, y, z) { return Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2) + Math.pow(z - 0.5, 2) < 0.1225 ? 1 : 0; },
  SHELL: function (x, y, z) { var r = Math.sqrt(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2) + Math.pow(z - 0.5, 2)); return r > 0.25 && r < 0.4 ? 1 : 0; },
  OCTANTS: function (x, y, z) { return ((x > 0.5 ? 1 : 0) + (y > 0.5 ? 1 : 0) + (z > 0.5 ? 1 : 0)) % 2; },
  CHECK3: function (x, y, z) { return (Math.min(2, Math.floor(x * 3)) + Math.min(2, Math.floor(y * 3)) + Math.min(2, Math.floor(z * 3))) % 2; },
  PLANE3D: function (x, y, z) { return x + y + z > 1.5 ? 1 : 0; },
  GAUSS3D: function (x, y, z) { return Math.exp(-(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2) + Math.pow(z - 0.5, 2)) / 0.06); },
  SINE3D: function (x, y, z) { return (Math.sin(2 * Math.PI * x) * Math.cos(Math.PI * y) * z) / 2 + 0.5; } };
var LAB_REG3 = ['GAUSS3D', 'SINE3D'];
function makeTask(name, opt) {
  opt = opt || {}; var density = Math.max(2, Math.round(opt.density || 9)), density2 = Math.max(2, Math.round(opt.density2d || opt.density || 9)), density3 = Math.max(2, Math.round(opt.density3d || 7)), r = rng((opt.seed || 1) + 77), xs = [], ys = [], kind = 'class', dims = 0;
  var sum = function (v) { return v.reduce(function (a, b) { return a + b; }, 0); };
  var logic = {
    AND: [2, function (v) { return [v[0] & v[1]]; }], OR: [2, function (v) { return [v[0] | v[1]]; }], NAND: [2, function (v) { return [1 - (v[0] & v[1])]; }],
    XOR: [2, function (v) { return [v[0] ^ v[1]]; }], COPY: [2, function (v) { return [v[0]]; }], NOT: [1, function (v) { return [1 - v[0]]; }],
    MAJ3: [3, function (v) { return [sum(v) >= 2 ? 1 : 0]; }], PAR3: [3, function (v) { return [sum(v) % 2]; }], MUX3: [3, function (v) { return [v[0] ? v[2] : v[1]]; }],
    CMP2: [4, function (v) { return [(v[0] + 2 * v[1]) > (v[2] + 2 * v[3]) ? 1 : 0]; }], MAJ5: [5, function (v) { return [sum(v) >= 3 ? 1 : 0]; }],
    PAR4: [4, function (v) { return [sum(v) % 2]; }], PAR5: [5, function (v) { return [sum(v) % 2]; }],
    ADD2: [4, function (v) { var s = v[0] + 2 * v[1] + v[2] + 2 * v[3]; return [s & 1, (s >> 1) & 1, (s >> 2) & 1]; }],
    DEC2: [2, function (v) { var k = v[0] + 2 * v[1], o = [0, 0, 0, 0]; o[k] = 1; return o; }],
    MAJ7: [7, function (v) { return [sum(v) >= 4 ? 1 : 0]; }], PAR6: [6, function (v) { return [sum(v) % 2]; }],
    CMP3: [6, function (v) { return [(v[0] + 2 * v[1] + 4 * v[2]) > (v[3] + 2 * v[4] + 4 * v[5]) ? 1 : 0]; }],
    ADD3: [3, function (v) { var s = sum(v); return [s & 1, (s >> 1) & 1]; }],
    MUL2: [4, function (v) { var p = (v[0] + 2 * v[1]) * (v[2] + 2 * v[3]); return [p & 1, (p >> 1) & 1, (p >> 2) & 1, (p >> 3) & 1]; }]
  };
  var segments = [[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1]]; // a b c d e f g of the digits 0 - 9
  var seq = false, layerInputs = 0, labels = [], classes = null; // labels: a short name of each sample, classes: the name of each output
  if (logic[name]) labBits(logic[name][0]).forEach(function (v) { xs.push(v); ys.push(logic[name][1](v)); });
  else if (name === 'SEG7') { for (var dg = 0; dg < 10; dg++) { xs.push([dg & 1, (dg >> 1) & 1, (dg >> 2) & 1, (dg >> 3) & 1]); ys.push(segments[dg].slice()); labels.push('digit ' + dg); } classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']; }
  else if (name === 'DIGITS') { // every digit, then noisy copies (1 to 3 pixels flipped)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    for (var d = 0; d < 10; d++) {
      var one = []; for (var o2 = 0; o2 < 10; o2++) one.push(o2 === d ? 1 : 0);
      for (var n2 = 0; n2 < 9; n2++) {
        var pix = labDigit(d), flips = n2 === 0 ? 0 : 1 + Math.floor(r() * 3);
        for (var f2 = 0; f2 < flips; f2++) { var at = Math.floor(r() * 35); pix[at] = 1 - pix[at]; }
        xs.push(pix); ys.push(one); labels.push('digit ' + d + (flips ? ', ' + flips + ' px flipped' : ''));
      }
    }
  } else if (name === 'WORDS') {
    classes = ['English', 'French'];
    ['en', 'fr'].forEach(function (lang, k) { LAB_WORDS[lang].split(' ').forEach(function (w) { if (w.length <= 6) { xs.push(labWord(w)); ys.push(k ? [0, 1] : [1, 0]); labels.push(w); } }); });
  } else if (name === 'MEMORY') { // a sequence of 5 steps of 1 bit: the brain reads one step at a time
    seq = true; layerInputs = 5;
    labBits(5).forEach(function (v) { xs.push(v.map(function (b) { return [b]; })); ys.push([v[0] ^ v[4]]); labels.push('bits ' + v.join('')); });
  } else if (name === 'TICTACTOE') { // legal boards (X plays first), as many of each winner (200 each, the same for a seed)
    var groups = [[], [], []];
    for (var code = 0; code < 19683; code++) {
      var board = [], c3 = code, nx = 0, no = 0;
      for (var cell = 0; cell < 9; cell++) { board.push(c3 % 3); c3 = Math.floor(c3 / 3); if (board[cell] === 1) nx++; if (board[cell] === 2) no++; }
      var win = labWinner(board);
      if (nx - no < 0 || nx - no > 1 || win === 3 || (win === 1 && nx !== no + 1) || (win === 2 && nx !== no)) continue;
      groups[win].push(board);
    }
    groups.forEach(function (list, wnr) {
      for (var t3 = 0; t3 < 200 && list.length; t3++) {
        var b3 = list.splice(Math.floor(r() * list.length), 1)[0];
        xs.push(labBoard(b3)); ys.push(wnr === 1 ? [1, 0, 0] : (wnr === 2 ? [0, 1, 0] : [0, 0, 1]));
        labels.push([0, 3, 6].map(function (row) { return b3.slice(row, row + 3).map(function (c) { return '.XO'[c]; }).join(''); }).join(' '));
      }
    });
    classes = ['X wins', 'O wins', 'nobody'];
  }
  else if (LAB_CURVES[name]) {
    dims = 1; kind = LAB_CLASS_CURVES.indexOf(name) >= 0 ? 'class' : 'reg';
    for (var k = 0; k < density; k++) { var x = k / (density - 1); xs.push([x]); ys.push([LAB_CURVES[name](x)]); }
  } else if (name === 'SPIRALS') {
    dims = 2;
    for (var c = 0; c < 2; c++) for (var p = 0; p < density2 * 2; p++) {
      var a = p / (density2 * 2) * 3 * Math.PI, rad = 0.05 + 0.4 * p / (density2 * 2), sx = c ? -1 : 1;
      xs.push([0.5 + sx * rad * Math.cos(a), 0.5 + sx * rad * Math.sin(a)]); ys.push([c]);
    }
  } else if (LAB_SOLIDS[name]) { // the 3D cubes: a grid of density3 ^ 3 points
    dims = 3; kind = LAB_REG3.indexOf(name) >= 0 ? 'reg' : 'class';
    for (var a3 = 0; a3 < density3; a3++) for (var b3 = 0; b3 < density3; b3++) for (var c3 = 0; c3 < density3; c3++) { var p3 = [a3 / (density3 - 1), b3 / (density3 - 1), c3 / (density3 - 1)]; xs.push(p3); ys.push([LAB_SOLIDS[name](p3[0], p3[1], p3[2])]); }
  } else { // the 2D grids
    dims = 2; kind = LAB_REG2.indexOf(name) >= 0 ? 'reg' : 'class';
    var g = LAB_SHAPES[name];
    for (var i = 0; i < density2; i++) for (var j = 0; j < density2; j++) { var gx = i / (density2 - 1), gy = j / (density2 - 1); xs.push([gx, gy]); ys.push([g(gx, gy)]); }
  }
  // the test set: a part of the samples never trained on (the same ones for a seed)
  var order = xs.map(function (_, i) { return i; }), testCount = Math.min(xs.length - 1, Math.round(xs.length * Math.max(0, Math.min(0.9, opt.test || 0))));
  for (var s = order.length - 1; s > 0; s--) { var q = Math.floor(r() * (s + 1)), tmp = order[s]; order[s] = order[q]; order[q] = tmp; }
  var testSet = order.slice(0, testCount).sort(function (a, b) { return a - b; }), isTest = {};
  testSet.forEach(function (i) { isTest[i] = true; });
  var task = { name: name, kind: kind, dims: dims, seq: seq, outputs: ys[0].length, inputs: seq ? xs[0][0].length : xs[0].length, layerInputs: layerInputs || (seq ? 0 : xs[0].length), in: [], out: [], clean: [], test: { in: [], out: [] }, all: { in: xs, out: ys, label: labels.length ? labels : null }, classes: classes, isTest: isTest };
  xs.forEach(function (x, i) {
    if (isTest[i]) { task.test.in.push(x); task.test.out.push(ys[i]); return; }
    var noisy = ys[i].map(function (v) { return opt.noise ? Math.min(1, Math.max(0, v + (r() * 2 - 1) * opt.noise)) : v; }); // the noise only on the trained targets
    task.in.push(x); task.out.push(noisy); task.clean.push(ys[i]);
  });
  return task;
}
if (typeof module !== 'undefined') module.exports = { makeTask: makeTask, LAB_TASKS: LAB_TASKS, LAB_TASK_INFO: LAB_TASK_INFO, labDigit: labDigit, labWord: labWord, labBoard: labBoard, labWinner: labWinner, LAB_PRESETS: LAB_PRESETS };

// ---------- evaluation shared by the page and the batch workers ----------
function labSampleError(y, t) { var e = 0; for (var k = 0; k < t.length; k++) e += Math.abs(y[k] - t[k]); return e / t.length; } // mean over the outputs
function labGood(task, y, t) { for (var k = 0; k < t.length; k++) { if (task.kind === 'reg' ? Math.abs(y[k] - t[k]) >= 0.1 : (y[k] >= 0.5) !== (t[k] >= 0.5)) return false; } return true; }
function labError(net, xs, ts) { if (!xs.length) return NaN; var e = 0; for (var i = 0; i < xs.length; i++) e += labSampleError(net.forward(xs[i]), ts[i]); return e / xs.length; }
function labSolved(task, net, xs, ts) { if (!xs.length) return null; for (var i = 0; i < xs.length; i++) if (!labGood(task, net.forward(xs[i]), ts[i])) return false; return true; }
// coefficient of the layered network (the error, positive = too high) and score of the brain one (positive = good), one per output
function labLayerCoef(mode, rate, y, t) { return y.map(function (v, k) { var d = v - t[k]; return mode === 'loss' ? rate * d * d : rate * d; }); }
function labBrainScore(mode, rate, y, t) { return y.map(function (v, k) { var e = t[k] - v; return mode === 'sign' ? rate * Math.sign(e) : (mode === 'loss' ? rate * e * e : rate * e); }); }
if (typeof module !== 'undefined') Object.assign(module.exports, { labSampleError: labSampleError, labGood: labGood, labError: labError, labSolved: labSolved, labLayerCoef: labLayerCoef, labBrainScore: labBrainScore });

// One round of a brain network: rate / (1 + decay * round), each sample applied at once or (batch) the sum of the round applied once
function labBrainRound(net, task, train, round) {
  var rate = train.rate / (1 + (train.decay || 0) * round), count = { added: 0, removed: 0, revived: 0 }, ev, i;
  if (net.opt.revive) net.watch();
  net.lrScale = 1 / (1 + (train.decay || 0) * round); // the decay applies to Adam too (its steps ignore the size of the score)
  var add = function (e) { count.added += e.added.length; count.removed += e.removed.length; };
  for (i = 0; i < task.in.length; i++) {
    var score = labBrainScore(train.coef, rate, net.forward(task.in[i]), task.out[i]);
    if (train.batch) net.accumulate(score.map(function (v) { return v / task.in.length; })); else { ev = net.learn(score); add(ev); }
  }
  if (train.batch) add(net.apply());
  if (net.opt.revive) count.revived = net.revive();
  return count;
}
if (typeof module !== 'undefined') module.exports.labBrainRound = labBrainRound;
