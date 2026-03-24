"""
algorithms.py
-------------
Pure-Python implementations of the algorithms used in the
Tourism Route Recommendation System.

Contents
--------
Sorting
  - insertion_sort(items)  : O(n²) simple insertion sort
  - merge_sort(items)      : O(n log n) divide-and-conquer sort

Searching
  - sequential_search(items, target) : O(n) linear scan
  - binary_search(items, target)     : O(log n) search on a SORTED list

Route Optimisation (TSP)
  - tsp_nearest_neighbor(graph, selected_nodes)
      Greedy nearest-neighbour heuristic for the Travelling Salesman Problem.
      Returns an ordered list of nodes and the total distance.
"""

import math
import networkx as nx


# ==========================================================================
# Sorting Algorithms
# ==========================================================================

def insertion_sort(items: list[str]) -> list[str]:
    """
    Sort a list of strings alphabetically using Insertion Sort.

    How it works
    ------------
    Start from the second element and, for each element, compare it
    backwards with the sorted portion on its left.  Shift larger elements
    one position to the right until we find the correct spot, then insert.

    Time complexity : O(n²)  — practical for small lists
    Space complexity: O(1)   — sorts in-place (we work on a copy)

    Parameters
    ----------
    items : list[str]
        The list of location names to sort.

    Returns
    -------
    list[str]
        A new sorted list (original is not modified).
    """
    arr = list(items)           # work on a copy so the original stays intact

    for i in range(1, len(arr)):
        # Pick the current element to be inserted into the sorted portion
        key = arr[i]
        j = i - 1

        # Move elements that are greater than `key` one step to the right
        while j >= 0 and arr[j].lower() > key.lower():
            arr[j + 1] = arr[j]
            j -= 1

        # Insert the key into its correct position
        arr[j + 1] = key

    return arr


def merge_sort(items: list[str]) -> list[str]:
    """
    Sort a list of strings alphabetically using Merge Sort.

    How it works
    ------------
    Recursively split the list in half until each sub-list has one element
    (which is trivially sorted), then merge pairs of sorted sub-lists back
    together in the right order.

    Time complexity : O(n log n) — efficient for larger lists
    Space complexity: O(n)       — needs extra space for the merge step

    Parameters
    ----------
    items : list[str]
        The list of location names to sort.

    Returns
    -------
    list[str]
        A new sorted list (original is not modified).
    """
    arr = list(items)           # work on a copy

    if len(arr) <= 1:
        return arr              # base case: a single element is already sorted

    # --- Divide ---
    mid = len(arr) // 2
    left_half  = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    # --- Conquer (merge) ---
    return _merge(left_half, right_half)


def _merge(left: list[str], right: list[str]) -> list[str]:
    """
    Helper for merge_sort.
    Merge two already-sorted lists into one sorted list.
    """
    merged = []
    i = j = 0

    # Compare elements from both halves and add the smaller one first
    while i < len(left) and j < len(right):
        if left[i].lower() <= right[j].lower():
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Append any remaining elements (only one of these loops will execute)
    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


# ==========================================================================
# Searching Algorithms
# ==========================================================================

def sequential_search(items: list[str], target: str) -> tuple[int, list[str]]:
    """
    Search for a location by name using Sequential (Linear) Search.

    Scans the list from left to right, comparing each element with the
    target.  Case-insensitive partial matching is supported.

    Time complexity : O(n)
    Space complexity: O(1)

    Parameters
    ----------
    items  : list[str]  — The list of location names to search.
    target : str        — The search query (substring, case-insensitive).

    Returns
    -------
    (index: int, matches: list[str])
        index   — index of the first exact match, or -1 if not found.
        matches — all items whose names contain the target substring.
    """
    target_lower = target.strip().lower()
    first_index  = -1
    matches      = []
    seen         = set()   # ป้องกันผลลัพธ์ซ้ำ

    for i, item in enumerate(items):
        if target_lower in item.lower() and item not in seen:  # partial, case-insensitive
            matches.append(item)
            seen.add(item)
            if first_index == -1:
                first_index = i

    return first_index, matches


def binary_search(sorted_items: list[str], target: str) -> tuple[int, list[str]]:
    """
    Search for a location by name using Binary Search.

    IMPORTANT: The input list must already be sorted alphabetically.
    Binary search only finds exact (prefix) matches because it relies on
    the ordering.  We perform a case-insensitive comparison.

    How it works
    ------------
    Compare the target with the middle element.
    - If equal   → found.
    - If target < middle → search the LEFT half.
    - If target > middle → search the RIGHT half.
    Repeat until found or the search space is empty.

    Time complexity : O(log n)  — much faster than sequential for large lists
    Space complexity: O(1)

    Parameters
    ----------
    sorted_items : list[str]  — Alphabetically sorted location names.
    target       : str        — Exact name to search for (case-insensitive).

    Returns
    -------
    (index: int, matches: list[str])
        index   — index in sorted_items where the match was found, or -1.
        matches — list containing the matched name, or empty list.
    """
    target_lower = target.strip().lower()
    low, high    = 0, len(sorted_items) - 1
    matches      = []

    while low <= high:
        mid = (low + high) // 2
        mid_lower = sorted_items[mid].lower()

        if mid_lower == target_lower:
            matches.append(sorted_items[mid])
            return mid, matches
        elif target_lower < mid_lower:
            high = mid - 1           # search left half
        else:
            low  = mid + 1           # search right half

    # Not found — fall back to a sequential scan for partial matches
    # so the UI can still show helpful suggestions
    _, partial_matches = sequential_search(sorted_items, target)
    return -1, partial_matches


# ==========================================================================
# TSP Route Optimisation — Nearest Neighbour Heuristic
# ==========================================================================

def tsp_nearest_neighbor(
    graph: nx.Graph,
    selected_nodes: list[str]
) -> tuple[list[str], float]:
    """
    Compute an approximate shortest travel route visiting all selected
    locations using the Nearest Neighbour (greedy) heuristic for the
    Travelling Salesman Problem (TSP).

    How it works
    ------------
    1. Start at the first selected location.
    2. At each step, move to the nearest unvisited location that is
       reachable (directly or via the shortest path in the graph).
    3. Repeat until all selected locations are visited.
    4. Return to the starting location to complete the tour.

    We use NetworkX's shortest_path_length to handle cases where two
    selected locations are not directly connected.

    Time complexity : O(n²)  where n = number of selected nodes.
    (Good enough for typical tourist route sizes.)

    Parameters
    ----------
    graph          : nx.Graph     — The full tourist-attraction graph.
    selected_nodes : list[str]    — Locations the user wants to visit.

    Returns
    -------
    (route: list[str], total_distance: float)
        route          — Ordered list of locations forming the tour.
                         Starts and ends at the same location.
        total_distance — Approximate total travel distance in km.
        
    Raises
    ------
    ValueError
        If fewer than 2 nodes are selected, or nodes are not reachable
        from each other in the graph.
    """
    if len(selected_nodes) < 2:
        raise ValueError("กรุณาเลือกสถานที่อย่างน้อย 2 แห่งเพื่อคำนวณเส้นทาง")

    # Build a distance matrix between all selected nodes using
    # NetworkX shortest path lengths (accounts for indirect routes)
    dist_matrix: dict[str, dict[str, float]] = {}
    for node in selected_nodes:
        dist_matrix[node] = {}
        for other in selected_nodes:
            if node == other:
                dist_matrix[node][other] = 0.0
            else:
                try:
                    dist_matrix[node][other] = nx.shortest_path_length(
                        graph, node, other, weight="weight"
                    )
                except nx.NetworkXNoPath:
                    # If no path exists, use a very large number (infinity)
                    dist_matrix[node][other] = math.inf

    # --- Nearest Neighbour greedy algorithm ---
    start_node   = selected_nodes[0]
    unvisited    = set(selected_nodes[1:])
    route        = [start_node]
    total_dist   = 0.0
    current      = start_node

    while unvisited:
        # Find the nearest unvisited node
        nearest      = None
        nearest_dist = math.inf

        for candidate in unvisited:
            d = dist_matrix[current][candidate]
            if d < nearest_dist:
                nearest_dist = d
                nearest      = candidate

        if nearest is None or nearest_dist == math.inf:
            raise ValueError(
                "สถานที่บางแห่งที่เลือกไม่มีเส้นทางเชื่อมต่อกันในกราฟ "
                "กรุณาเพิ่มเส้นทางระหว่างสถานที่ หรือเลือกสถานที่ใหม่"
            )

        # Move to the nearest unvisited location
        route.append(nearest)
        total_dist += nearest_dist
        unvisited.remove(nearest)
        current = nearest

    # Return to the starting location to complete the round trip
    return_dist = dist_matrix[current][start_node]
    if return_dist == math.inf:
        raise ValueError(
            f"ไม่สามารถเดินทางกลับจาก '{current}' ไป '{start_node}' ได้ "
            "เนื่องจากไม่มีเส้นทางเชื่อมต่อ"
        )

    route.append(start_node)
    total_dist += return_dist

    return route, round(total_dist, 2)


def get_full_route_edges(
    graph: nx.Graph,
    route: list[str]
) -> list[tuple[str, str, float]]:
    """
    Given an ordered route (as returned by tsp_nearest_neighbor), expand
    each hop into the actual shortest-path edges so we can highlight them
    on the visualisation.

    Parameters
    ----------
    graph : nx.Graph
        The full tourist-attraction graph.
    route : list[str]
        Ordered list of locations (including the return to start).

    Returns
    -------
    list of (node_a, node_b, distance) tuples
        representing every individual edge that forms the optimal route.
    """
    route_edges = []
    for i in range(len(route) - 1):
        src, dst = route[i], route[i + 1]
        try:
            path = nx.shortest_path(graph, src, dst, weight="weight")
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                d = graph[u][v]["weight"]
                route_edges.append((u, v, d))
        except nx.NetworkXNoPath:
            pass   # skip unreachable hops (shouldn't happen after validation)

    return route_edges
