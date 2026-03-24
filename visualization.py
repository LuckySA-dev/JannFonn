"""
visualization.py
----------------
Draws the tourist-attraction graph using Matplotlib.

Features
--------
- Isolated nodes (no edges) are placed in a neat row at the bottom,
  separate from the spring-layout cluster.
- When an optimal route is provided, those edges are highlighted in red.
- When search_matches is provided, those nodes are highlighted in crimson.
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import networkx as nx
from algorithms import get_full_route_edges


# ---------------------------------------------------------------------------
# Thai font detection
# ---------------------------------------------------------------------------

def _get_thai_font() -> str:
    preferred = [
        # Windows
        "Leelawadee UI", "Leelawadee", "Tahoma", "Arial Unicode MS",
        # Linux — fonts-thai-tlwg (Streamlit Cloud / Ubuntu)
        "Waree", "Garuda", "Norasi", "Laksaman", "Umpush", "Kinnari",
        # Linux — fonts-noto (if installed)
        "Noto Sans Thai",
    ]
    available = {f.name for f in fm.fontManager.ttflist}

    # ถ้ายังไม่พบฟอนต์ไทยเลย ลอง rebuild font cache
    # (Streamlit Cloud: fonts ถูกติดตั้งจาก packages.txt ก่อน process เริ่ม
    #  แต่ matplotlib cache อาจยังไม่รู้จัก)
    if not any(f in available for f in preferred):
        try:
            fm.fontManager = fm.FontManager()
            available = {f.name for f in fm.fontManager.ttflist}
        except Exception:
            pass

    for font in preferred:
        if font in available:
            return font
    return "DejaVu Sans"


_THAI_FONT = _get_thai_font()
matplotlib.rcParams["font.family"] = _THAI_FONT


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLOUR_NODE_NORMAL  = "#4A90D9"   # blue   — ordinary nodes
COLOUR_NODE_ROUTE   = "#E67E22"   # orange — route nodes
COLOUR_NODE_START   = "#27AE60"   # green  — start/end node
COLOUR_NODE_SEARCH  = "#E53935"   # red    — search-result nodes
COLOUR_EDGE_NORMAL  = "#B0BEC5"   # grey   — normal edges
COLOUR_EDGE_ROUTE   = "#E74C3C"   # red    — highlighted route edges
COLOUR_WEIGHT       = "#37474F"   # dark grey — edge weight labels


# ---------------------------------------------------------------------------
# Layout helper — keeps isolated nodes out of the spring cluster
# ---------------------------------------------------------------------------

def _compute_layout(graph: nx.Graph, seed: int, k_value: float) -> dict:
    """
    Position nodes using spring_layout, but place isolated nodes
    (degree = 0) in a tidy row just below the main cluster.
    """
    if graph.number_of_nodes() == 0:
        return {}

    isolated  = [n for n in graph.nodes() if graph.degree(n) == 0]
    connected = [n for n in graph.nodes() if graph.degree(n) > 0]

    pos = {}

    if connected:
        sub     = graph.subgraph(connected)
        sub_pos = nx.spring_layout(sub, seed=seed, k=k_value)
        pos.update(sub_pos)

    if isolated:
        if pos:
            # Place isolated nodes below the lowest connected node
            min_y = min(p[1] for p in pos.values()) - 0.50
        else:
            min_y = 0.0

        step = 2.0 / max(len(isolated), 1)
        for i, node in enumerate(isolated):
            pos[node] = (-1.0 + i * step + step / 2, min_y)

    return pos


# ---------------------------------------------------------------------------
# Public drawing function
# ---------------------------------------------------------------------------

def draw_graph(
    graph: nx.Graph,
    optimal_route: list[str] | None = None,
    search_matches: list[str] | None = None,
    figsize: tuple[int, int] = (10, 7),
    seed: int = 42,
) -> plt.Figure:
    """
    Draw the tourist-attraction graph and return a Matplotlib Figure.

    Parameters
    ----------
    graph         : nx.Graph
    optimal_route : ordered route list from tsp_nearest_neighbor (optional)
    search_matches: list of node names to highlight in red (optional)
    figsize       : (width, height) in inches
    seed          : random seed for reproducible layout
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F0F4F8")
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_title(
        "กราฟแสดงสถานที่ท่องเที่ยว",
        fontsize=16, fontweight="bold", color="#1A237E", pad=15,
        fontfamily=_THAI_FONT,
    )
    ax.axis("off")

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5,
            "ยังไม่มีสถานที่\nกรุณาเพิ่มสถานที่ท่องเที่ยวจากแถบด้านข้าง",
            ha="center", va="center", fontsize=13, color="#546E7A",
            transform=ax.transAxes, fontfamily=_THAI_FONT,
        )
        return fig

    # ---- Layout -----------------------------------------------------------
    k_value = 2.0 / math.sqrt(graph.number_of_nodes()) if graph.number_of_nodes() > 1 else 1.0
    pos = _compute_layout(graph, seed, k_value)

    # ---- Route sets -------------------------------------------------------
    route_edge_set: set[frozenset] = set()
    route_node_set: set[str]       = set()
    start_node: str | None         = None

    if optimal_route and len(optimal_route) >= 2:
        for u, v, _ in get_full_route_edges(graph, optimal_route):
            route_edge_set.add(frozenset([u, v]))
        route_node_set = set(optimal_route[:-1])
        start_node     = optimal_route[0]

    # ---- Search set -------------------------------------------------------
    search_set: set[str] = set(search_matches) if search_matches else set()

    # ---- Classify edges ---------------------------------------------------
    normal_edges      = []
    highlighted_edges = []
    for u, v in graph.edges():
        if frozenset([u, v]) in route_edge_set:
            highlighted_edges.append((u, v))
        else:
            normal_edges.append((u, v))

    # ---- Draw edges -------------------------------------------------------
    nx.draw_networkx_edges(
        graph, pos, edgelist=normal_edges,
        edge_color=COLOUR_EDGE_NORMAL, width=2.0, alpha=0.8, ax=ax,
    )
    if highlighted_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=highlighted_edges,
            edge_color=COLOUR_EDGE_ROUTE, width=4.0, alpha=0.95, ax=ax,
        )

    # ---- Classify nodes ---------------------------------------------------
    # Priority: start > route > search > normal
    route_only  = [n for n in route_node_set if n != start_node]
    search_only = [n for n in search_set if n not in route_node_set and n != start_node]
    normal_only = [
        n for n in graph.nodes()
        if n not in route_node_set and n != start_node and n not in search_set
    ]

    # ---- Draw nodes -------------------------------------------------------
    nx.draw_networkx_nodes(
        graph, pos, nodelist=normal_only,
        node_color=COLOUR_NODE_NORMAL, node_size=900, ax=ax,
    )
    if search_only:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=search_only,
            node_color=COLOUR_NODE_SEARCH, node_size=1050, ax=ax,
        )
    if route_only:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=route_only,
            node_color=COLOUR_NODE_ROUTE, node_size=1000, ax=ax,
        )
    if start_node and start_node in graph.nodes():
        nx.draw_networkx_nodes(
            graph, pos, nodelist=[start_node],
            node_color=COLOUR_NODE_START, node_size=1100, ax=ax,
        )

    # ---- Node labels (below each node) ------------------------------------
    for node, (x, y) in pos.items():
        if node == start_node:
            bg_color = COLOUR_NODE_START
        elif node in route_node_set:
            bg_color = COLOUR_NODE_ROUTE
        elif node in search_set:
            bg_color = COLOUR_NODE_SEARCH
        else:
            bg_color = COLOUR_NODE_NORMAL

        ax.text(
            x, y - 0.14, node,
            ha="center", va="top",
            fontsize=8, fontweight="bold",
            fontfamily=_THAI_FONT, color="#1A237E",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white",
                ec=bg_color, linewidth=1.5, alpha=0.92,
            ),
            zorder=5,
        )

    # ---- Edge weight labels -----------------------------------------------
    edge_labels = nx.get_edge_attributes(graph, "weight")
    formatted_labels = {
        (u, v): (f"{int(w)} กม." if w == int(w) else f"{w:.1f} กม.")
        for (u, v), w in edge_labels.items()
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=formatted_labels,
        font_size=8, font_color=COLOUR_WEIGHT, font_family=_THAI_FONT,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        ax=ax,
    )

    # ---- Route order numbers ----------------------------------------------
    if optimal_route and len(optimal_route) >= 2:
        for step, node in enumerate(optimal_route[:-1], start=1):
            if node in pos:
                x, y = pos[node]
                ax.text(
                    x, y + 0.07, f"#{step}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="#C0392B", ec="none"),
                    zorder=6,
                )

    # ---- Legend -----------------------------------------------------------
    handles = [
        mpatches.Patch(color=COLOUR_NODE_NORMAL, label="สถานที่ท่องเที่ยว"),
        mpatches.Patch(color=COLOUR_EDGE_NORMAL, label="เส้นทางการเดินทาง"),
    ]
    if search_matches:
        handles.append(mpatches.Patch(color=COLOUR_NODE_SEARCH, label="ผลการค้นหา"))
    if optimal_route:
        handles += [
            mpatches.Patch(color=COLOUR_NODE_START, label="จุดเริ่มต้น / สิ้นสุด"),
            mpatches.Patch(color=COLOUR_NODE_ROUTE, label="สถานที่ในเส้นทาง"),
            mpatches.Patch(color=COLOUR_EDGE_ROUTE, label="เส้นทางที่ดีที่สุด"),
        ]

    ax.legend(
        handles=handles, loc="upper left", framealpha=0.9,
        fontsize=9, title="คำอธิบาย", title_fontsize=10,
        prop={"family": _THAI_FONT, "size": 9},
    )

    plt.tight_layout()
    return fig
