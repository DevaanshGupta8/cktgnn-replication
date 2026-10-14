"""Open-source AC simulator for OCB behavioural op-amp circuits.

Re-implements the netlist semantics of external/CktGNN/OCB/src/simulator/graph_to_fom.py
(Cadence Spectre + Ocean in the original) with a small modified-nodal-analysis solver,
so that OCB circuits can be evaluated without a commercial simulator.

Element templates (graph_to_fom.py, `subnetlist_node`):
  * every gm subgraph: VCCS  G0 (OUT GND IN GND) gm=+-x  plus R0=1M and C0=50f from OUT to GND
  * R/C in parallel with a gm stage sit between IN and OUT; series R-C uses an internal node
  * passive subgraphs (types 2-5) are two-terminal IN-OUT elements
  * feedback-direction types ('-' in direction_node) swap the IN and OUT nets
Value scaling (basis_node/unit_node): R = 0.1*v MOhm, C = 10*v fF, gm = 0.1*v mS, v in 1..101.
Measurements (ocean.txt): AC 1 Hz..100 GHz at 100 points/decade,
gain = dB20(H) at the first point, PM via phaseMargin(H), BW = unityGainFreq(H).
"""
import numpy as np

R_SCALE, C_SCALE, GM_SCALE = 1e5, 1e-14, 1e-4
R_PAR, C_PAR = 1e6, 50e-15                      # fixed output parasitics of every gm stage
FREQS = np.logspace(0, 11, 11 * 100 + 1)        # ocean.txt: ac 1 Hz .. 100 GHz, 100/decade

GM_TYPES = set(range(6, 26))
NEG_GM_TYPES = {7, 9, 11, 13, 15, 17, 19, 21, 23, 25}
FEEDBACK_TYPES = {8, 9, 12, 13, 16, 17, 20, 21, 24, 25}
PAR_C_TYPES = set(range(10, 14)) | set(range(18, 22))   # C1 between IN and OUT
PAR_R_TYPES = set(range(14, 22))                         # R1 between IN and OUT
SERIES_RC_TYPES = set(range(22, 26))                     # R1 IN-x, C1 x-OUT


def circuit_elements(g, gm_sign=-1.0, pos7_to_ground=False):
    """Turn a CktGNN subgraph-level igraph into a list of elements.

    Returns (elements, node_names). Elements are tuples
    ('Y', a, b, g_real, c_imag) for an admittance g + jwC between nets a and b, or
    ('G', out, inp, gm) for a VCCS injecting gm*V(inp) INTO net `out`.
    Net 'gnd' is ground, 'vin' is the ideal AC source.
    gm_sign=-1 encodes the SPICE/Spectre convention (current gm*V(IN) leaves OUT),
    i.e. a positive gm stage is inverting.
    """
    pos_of = {v.index: v['pos'] for v in g.vs}
    by_pos = {v['pos']: v for v in g.vs}
    main = [p for p in (2, 3, 4) if p in by_pos]
    out_net = {}
    for v in g.vs:
        if v['type'] == 0:
            out_net[v.index] = 'vin'
        elif v['type'] == 1:
            out_net[v.index] = 'out'
    for k, p in enumerate(main):
        out_net[by_pos[p].index] = 'out' if k == len(main) - 1 else f'n{k + 1}'

    elements, internal = [], 0
    for v in g.vs:
        t = v['type']
        if t in (0, 1):
            continue
        if pos_of[v.index] in main:
            k = main.index(pos_of[v.index])
            a = 'vin' if k == 0 else f'n{k}'
            b = out_net[v.index]
        else:
            preds, succs = g.predecessors(v.index), g.successors(v.index)
            assert len(preds) == 1 and len(succs) == 1, (t, preds, succs)
            a, b = out_net[preds[0]], out_net[succs[0]]
            if pos7_to_ground and v['pos'] == 7:
                b = 'gnd'
        if t in FEEDBACK_TYPES:
            a, b = b, a
        r, c, gm = v['r'] * R_SCALE, v['c'] * C_SCALE, v['gm'] * GM_SCALE
        if t == 2:
            elements.append(('Y', a, b, 1 / r, 0.0))
        elif t == 3:
            elements.append(('Y', a, b, 0.0, c))
        elif t == 4:                       # R (a-x) in series with C (x-b)
            x = f'x{internal}'; internal += 1
            elements += [('Y', a, x, 1 / r, 0.0), ('Y', x, b, 0.0, c)]
        elif t == 5:
            elements += [('Y', a, b, 1 / r, 0.0), ('Y', a, b, 0.0, c)]
        if t in GM_TYPES:
            sign = -1.0 if t in NEG_GM_TYPES else 1.0
            elements += [('G', b, a, gm_sign * sign * gm),
                         ('Y', b, 'gnd', 1 / R_PAR, 0.0), ('Y', b, 'gnd', 0.0, C_PAR)]
            if t in PAR_C_TYPES:
                elements.append(('Y', a, b, 0.0, c))
            if t in PAR_R_TYPES:
                elements.append(('Y', a, b, 1 / r, 0.0))
            if t in SERIES_RC_TYPES:
                x = f'x{internal}'; internal += 1
                elements += [('Y', a, x, 1 / r, 0.0), ('Y', x, b, 0.0, c)]
    nodes = sorted({n for e in elements for n in e[1:3]} - {'gnd', 'vin'})
    return elements, nodes


def mna(g, **kw):
    """MNA matrices of the AC circuit: (G + sC) V = (bG + s bC) * V(vin); also returns the output index."""
    elements, nodes = circuit_elements(g, **kw)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    G = np.zeros((n, n)); C = np.zeros((n, n))
    bG = np.zeros(n); bC = np.zeros(n)            # contributions of the 1 V source column
    for e in elements:
        if e[0] == 'Y':
            _, a, b, gr, cj = e
            for (p, q, s) in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                if p in idx:
                    if q in idx:
                        G[idx[p], idx[q]] += s * gr; C[idx[p], idx[q]] += s * cj
                    elif q == 'vin':
                        bG[idx[p]] -= s * gr; bC[idx[p]] -= s * cj
        else:                                       # VCCS: gm*V(inp) injected into `out`
            _, out, inp, gm = e
            if out in idx:                          # KCL (sum leaving = 0): -gm*V(inp) on LHS
                if inp in idx:
                    G[idx[out], idx[inp]] -= gm
                elif inp == 'vin':
                    bG[idx[out]] += gm
    return G, C, bG, bC, idx['out']


def transfer_function(g, freqs=FREQS, **kw):
    """H(jw) = V(out)/V(vin) on the frequency grid (complex array)."""
    G, C, bG, bC, o = mna(g, **kw)
    w = 2 * np.pi * freqs
    Y = G[None] + 1j * w[:, None, None] * C[None]
    rhs = (bG[None] + 1j * w[:, None] * bC[None])[..., None]
    V = np.linalg.solve(Y, rhs)[..., 0]
    return V[:, o]


def _finite_poles(A, B):
    from scipy.linalg import eigvals
    with np.errstate(all='ignore'):
        p = eigvals(A, B)
    return p[np.isfinite(p)]


def stability(g, rel_tol=1e-9, **kw):
    """Pole-based stability (exact for these linear circuits).
    Open loop: poles of det(G + sC) = 0.
    Closed loop: the amplifier as a unity-gain buffer with negative feedback, V(vin) = -k V(out) with
    k = sign(A(0)) so that the DC loop gain k*A(0) is positive. The loop is stable iff every finite pole
    has a negative real part."""
    G, C, bG, bC, o = mna(g, **kw)
    a0 = np.linalg.solve(G, bG)[o]                 # DC gain A(0) (real)
    k = 1.0 if a0 >= 0 else -1.0
    e = np.zeros(len(bG)); e[o] = 1.0
    p_ol = _finite_poles(G, -C)
    p_cl = _finite_poles(G + k * np.outer(bG, e), -(C + k * np.outer(bC, e)))
    def worst(p):
        return float(np.max(p.real / np.maximum(1.0, np.abs(p)))) if len(p) else -1.0
    return {'dc_gain': float(a0), 'ol_stable': worst(p_ol) < rel_tol, 'cl_stable': worst(p_cl) < rel_tol,
            'ol_worst_rel_re': worst(p_ol), 'cl_worst_rel_re': worst(p_cl)}


def analyze(g, **kw):
    """Everything the optimisation study needs for one circuit: gain (dB), UGF (Hz), lag-based phase
    margin (deg; 180 minus phase lag accumulated from DC to the first 0 dB crossing, i.e. the unity-feedback
    PM when the DC loop gain is positive), number of 0 dB crossings, and pole-based stability."""
    H = transfer_function(g, **kw)
    m = measure(H)
    mag_db = 20 * np.log10(np.abs(H))
    m['n_crossings'] = int(np.sum(np.sign(mag_db[:-1]) != np.sign(mag_db[1:])))
    m['pm_lag'] = 180 - (m['phase_dc'] - m['phase_ugf_unwrapped']) if np.isfinite(m['ugf_hz']) else np.nan
    m.update(stability(g, **kw))
    return m


def measure(H, freqs=FREQS):
    """Gain (dB at first point), unity-gain frequency (Hz, first 0 dB down-crossing,
    log-f interpolation) and phase at UGF (deg, wrapped and unwrapped).
    Returns NaNs for UGF/phase when |H| never crosses 0 dB (Ocean reports a failure)."""
    mag_db = 20 * np.log10(np.abs(H))
    ph_unwrapped = np.degrees(np.unwrap(np.angle(H)))
    res = {'gain_db': mag_db[0], 'ugf_hz': np.nan, 'phase_ugf_wrapped': np.nan, 'phase_ugf_unwrapped': np.nan,
           'phase_dc': np.degrees(np.angle(H[0]))}
    cross = np.where((mag_db[:-1] >= 0) & (mag_db[1:] < 0))[0]
    if len(cross):
        i = cross[0]
        lf = np.log10(freqs)
        t = mag_db[i] / (mag_db[i] - mag_db[i + 1])
        res['ugf_hz'] = 10 ** (lf[i] + t * (lf[i + 1] - lf[i]))
        pu = ph_unwrapped[i] + t * (ph_unwrapped[i + 1] - ph_unwrapped[i])
        res['phase_ugf_unwrapped'] = pu
        res['phase_ugf_wrapped'] = (pu + 180) % 360 - 180
    return res


def fom_from_measurements(gain_db, pm_deg, ugf_hz):
    """FoM exactly as in graph_to_fom.py line 660."""
    return 1.2 * abs(gain_db) / 100 + 1.6 * pm_deg / (-90) + 10 * abs(ugf_hz) / 1e9
