# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 2026

@author: jamil

Standard diagnostics for a metabolic network (w, adjacency) as used
throughout the influx_diversity investigation scripts: a connectivity check
(is every resource actually reachable from the sole externally-supplied
resource?) and a networkx visualisation, plus a network sampler that
GUARANTEES full connectivity while staying sparse.

Motivated by a concrete finding: the M=25 sparse gamma-network investigation
(mean=0.04, variance=0.0014, seed 900201) turned out to have 4 disconnected
components, with the dominant (highest-energy, externally-supplied)
resource isolated in a 2-node component {9, 12} - the other 21 resources
could NEVER receive anything from the supply, no matter what the community
dynamics did, since there was no directed path from the dominant resource
to them at all. That's a structural property of the network, not a
dynamical outcome, and every M25_sparse_*.py script that used
sample_shared_network_gamma() without checking this was silently subject to
it. check_connectivity() below is meant to be called (and its result
inspected) right after sampling any network, before running simulations on
it, in every future investigation script.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gamma as gamma_dist


def check_connectivity(w, adjacency, verbose=True):

    '''

    Diagnose whether every resource is reachable from the dominant
    (highest-energy, externally-supplied) resource via directed edges of
    the metabolic network - the property that actually matters for a
    single-resource-supply investigation, which is STRONGER than simple
    (undirected/weak) connectedness. Even within the dominant resource's
    own weakly-connected component, some nodes can be unreachable from it:
    e.g. three resources A (w=0.9), B (w=0.5), C (w=0.3) with edges A->C
    and B->C is weakly connected (A-C-B all linked via C) and gated
    (every edge respects w_source > w_target), but B is NOT reachable from
    A - there is no directed path, only two separate paths converging on
    C. So this function checks true directed reachability from the
    dominant resource (via nx.descendants), not just which weakly-connected
    component a node falls in.

    Parameters
    ----------
    w : (M,) array of resource energies.
    adjacency : (M,M) 0/1 array, q_{alpha,beta}.
    verbose : bool - if True, prints a human-readable summary.

    Returns
    -------
    dict with keys:
        'dominant' : int, index of the highest-energy resource.
        'n_components' : int, number of weakly-connected components.
        'component_sizes' : sorted list of component sizes (largest first).
        'reachable_from_dominant' : sorted list of node indices reachable
            from the dominant resource via directed edges (including
            itself).
        'unreachable' : sorted list of node indices NOT reachable from the
            dominant resource - these can only ever hold whatever their
            initial condition gives them, decaying from there, regardless
            of community dynamics.
        'is_fully_connected' : bool, True iff every resource is reachable
            from the dominant resource (i.e. unreachable is empty).

    '''

    M = len(w)
    dominant = int(np.argmax(w))

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    edges = np.argwhere(adjacency > 0)
    G.add_edges_from([(int(a), int(b)) for a, b in edges])

    components = [sorted(c) for c in nx.weakly_connected_components(G)]
    components.sort(key=len, reverse=True)

    reachable = sorted(nx.descendants(G, dominant) | {dominant})
    unreachable = sorted(set(range(M)) - set(reachable))

    result = {
        'dominant': dominant,
        'n_components': len(components),
        'component_sizes': [len(c) for c in components],
        'reachable_from_dominant': reachable,
        'unreachable': unreachable,
        'is_fully_connected': len(unreachable) == 0,
    }

    if verbose:
        print(f"Network connectivity check (M={M}, dominant resource={dominant}):")
        print(f"  {result['n_components']} weakly-connected component(s), "
             f"sizes={result['component_sizes']}")
        if result['is_fully_connected']:
            print(f"  OK: all {M} resources reachable from the dominant resource.")
        else:
            print(f"  WARNING: {len(unreachable)}/{M} resources are NOT reachable "
                 f"from the dominant resource and can only ever decay from their "
                 f"initial condition: {unreachable}")

    return result


def plot_metabolic_network(w, adjacency, R_final=None, dominant=None,
                           title=None, save_path=None, ax=None):

    '''

    Plot the metabolic network q_{alpha,beta} with networkx - nodes
    positioned by resource energy (y-axis, higher energy at top, matching
    the gated a->b (w_a > w_b) direction of every edge), coloured/sized by
    R_final if supplied (log10 scale, useful for visually confirming which
    resources actually accumulate vs. which stay near the numerical floor -
    see check_connectivity() for WHY that happens structurally), otherwise
    a uniform colour. The edge(s) leaving the dominant resource are
    highlighted in red.

    Parameters
    ----------
    w : (M,) array of resource energies.
    adjacency : (M,M) 0/1 array, q_{alpha,beta}.
    R_final : (M,) array or None - final resource abundances to colour/size
        nodes by (log10 scale, floored at 1e-4 to avoid log(0)).
    dominant : int or None - index of the dominant resource; if None,
        inferred as argmax(w).
    title : str or None.
    save_path : str or None - if given, saves the figure to this path.
    ax : matplotlib Axes or None - if given, draws into it instead of
        creating a new figure (save_path is ignored in that case).

    Returns
    -------
    (fig, ax) if ax was None, else just ax.

    '''

    M = len(w)
    if dominant is None:
        dominant = int(np.argmax(w))

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    edges = np.argwhere(adjacency > 0)
    G.add_edges_from([(int(a), int(b)) for a, b in edges])

    order = np.argsort(-w)
    rank = {node: r for r, node in enumerate(order)}
    rng = np.random.RandomState(7)
    pos = {i: (((rank[i] % 5) - 2) * 0.9 + rng.uniform(-0.25, 0.25), w[i]) for i in range(M)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=150)

    cmap = LinearSegmentedColormap.from_list('seq_blue', ['#dbe6f0', '#2a78d6', '#0d2f52'])

    if R_final is not None:
        abundance = np.maximum(R_final, 1e-4)
        log_ab = np.log10(abundance)
        vmin, vmax = np.log10(1e-4), max(np.log10(max(abundance.max(), 1)), np.log10(1e-4) + 1e-6)
        node_colors = [cmap((log_ab[i] - vmin) / (vmax - vmin)) for i in range(M)]
        node_sizes = [260 + 1400 * ((log_ab[i] - vmin) / (vmax - vmin)) ** 2 for i in range(M)]
    else:
        node_colors = ['#2a78d6'] * M
        node_sizes = [320] * M

    edge_colors = ['#e34948' if (u == dominant or v == dominant) else '#9aa3a1'
                  for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True,
                           arrowsize=13, arrowstyle='-|>', connectionstyle='arc3,rad=0.05',
                           width=1.3, node_size=node_sizes)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                           edgecolors='#1a1d1f', linewidths=0.8)

    if R_final is not None:
        light_labels = {i: str(i) for i in range(M) if log_ab[i] > vmin + 0.5}
        dark_labels = {i: str(i) for i in range(M) if log_ab[i] <= vmin + 0.5}
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color='white', labels=light_labels)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color='#1a1d1f', labels=dark_labels)
    else:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color='white')

    ax.set_ylabel('Resource energy $w_a$')
    ax.set_xticks([])
    ax.set_title(title or f'Metabolic network $q_{{ab}}$ (M={M})')

    if R_final is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        (ax.figure).colorbar(sm, ax=ax, label='log10(final resource abundance)')

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        return fig, ax

    return ax


def sample_connected_gamma_network(M, mean, variance, seed, extra_edge_scale=0.3):

    '''

    Sample a metabolic network that is GUARANTEED to have every resource
    reachable from the dominant (highest-energy) resource via directed
    edges, while staying sparse - unlike sample_shared_network_gamma(),
    which independently Bernoulli-samples every possible edge and can (does,
    in practice - see this module's docstring) leave most of the network
    disconnected from the sole supply point.

    Construction: sort resources by energy descending (rank 0 = dominant).
    Every non-dominant resource (rank k >= 1) is assigned exactly one
    "parent" edge from a single, randomly chosen HIGHER-energy resource
    (rank 0..k-1) - chosen with probability proportional to the same
    gamma-distribution energy-gap preference used in
    sample_shared_network_gamma() (favouring small energy gaps), but as a
    genuine probability distribution over the available parents (not
    independent Bernoulli trials), so exactly one parent is always chosen.
    This alone is a spanning tree rooted at the dominant resource - M-1
    edges, every node reachable from the root by construction, no
    independent-sampling risk of disconnection.

    A small number of EXTRA edges are then layered on top (same gamma
    link-probability formula as sample_shared_network_gamma(), scaled down
    by extra_edge_scale) for occasional branching/redundancy, without
    threatening connectivity (removing extra edges can only ever reduce the
    graph back to the guaranteed-connected spanning tree, never below it).

    Parameters
    ----------
    M : int - number of resources.
    mean, variance : float - gamma distribution parameters for the energy-
        gap preference (same meaning/constraints as
        sample_shared_network_gamma() - requires mean**2/variance >= 1).
    seed : int.
    extra_edge_scale : float in [0, 1] - multiplies the gamma link
        probability used for extra (non-tree) edges. 0 gives the bare
        spanning tree (sparsest possible connected network, M-1 edges);
        larger values add progressively more redundant edges. Default 0.3
        chosen empirically to keep the result sparse (see this module's
        __main__ block for a verification sweep).

    Returns
    -------
    w : (M,) array of resource energies, Uniform(0,1).
    adjacency : (M,M) 0/1 array, q_{alpha,beta} - guaranteed weakly
        connected with every node reachable from argmax(w).

    '''

    if not (0 <= extra_edge_scale <= 1):
        raise ValueError(f"extra_edge_scale must be in [0, 1], got {extra_edge_scale}")

    rng = np.random.RandomState(seed)
    w = rng.uniform(0, 1, M)

    gamma_shape, gamma_scale = mean**2 / variance, variance / mean
    if gamma_shape < 1:
        raise ValueError(
            "sample_connected_gamma_network requires mean**2/variance >= 1 "
            f"(got shape={gamma_shape:.4g} from mean={mean}, variance={variance}). "
            "See sample_shared_network_gamma()'s docstring for why shape < 1 "
            "is degenerate.")
    mode = (gamma_shape - 1) * gamma_scale
    pdf_mode = gamma_dist.pdf(mode, a=gamma_shape, scale=gamma_scale)

    order = np.argsort(-w)  # rank 0 = dominant (highest energy)

    adjacency = np.zeros((M, M), dtype=int)

    # --- spanning tree: every non-root node picks exactly one higher-energy parent ---
    for k in range(1, M):
        node = order[k]
        parent_candidates = order[:k]
        gaps = w[parent_candidates] - w[node]
        weights = gamma_dist.pdf(gaps, a=gamma_shape, scale=gamma_scale) / pdf_mode
        weights = weights / weights.sum()
        parent = rng.choice(parent_candidates, p=weights)
        adjacency[parent, node] = 1

    # --- extra sparse edges on top, same gamma formula, scaled down ---
    energy_differences = w[:, np.newaxis] - w[np.newaxis, :]
    link_probability = gamma_dist.pdf(energy_differences, a=gamma_shape, scale=gamma_scale) / pdf_mode
    link_probability = link_probability * extra_edge_scale
    extra = rng.binomial(1, link_probability)
    adjacency = np.maximum(adjacency, extra)

    return w, adjacency


if __name__ == '__main__':

    # verification sweep: confirm the connected sampler is (a) always fully
    # connected and (b) stays sparse, across several seeds and
    # extra_edge_scale choices, compared against the plain (independently-
    # sampled, not-guaranteed-connected) gamma method at the same mean/variance
    import sys
    import os

    file_directory_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, file_directory_name)
    from timeout_utils import sample_shared_network_gamma

    M = 25
    mean, variance = 0.04, 0.0014
    seeds = range(900201, 900211)

    print("=== plain sample_shared_network_gamma (not guaranteed connected) ===")
    n_disconnected = 0
    for seed in seeds:
        w, adjacency = sample_shared_network_gamma(M, mean, variance, seed)
        result = check_connectivity(w, adjacency, verbose=False)
        if not result['is_fully_connected']:
            n_disconnected += 1
        print(f"  seed={seed}: components={result['n_components']}, "
             f"sizes={result['component_sizes']}, edges={int(adjacency.sum())}")
    print(f"{n_disconnected}/{len(list(seeds))} seeds disconnected from dominant resource")

    print()
    print("=== sample_connected_gamma_network, extra_edge_scale sweep ===")
    for extra_edge_scale in [0.0, 0.15, 0.3, 0.5]:
        edge_counts = []
        for seed in seeds:
            w, adjacency = sample_connected_gamma_network(M, mean, variance, seed,
                                                           extra_edge_scale=extra_edge_scale)
            result = check_connectivity(w, adjacency, verbose=False)
            assert result['is_fully_connected'], \
                f"extra_edge_scale={extra_edge_scale}, seed={seed}: NOT fully connected!"
            edge_counts.append(int(adjacency.sum()))
        print(f"  extra_edge_scale={extra_edge_scale}: all {len(list(seeds))} seeds fully "
             f"connected, edges={edge_counts} (mean={np.mean(edge_counts):.1f}, "
             f"spanning-tree minimum={M-1})")
