"""
graph_manager.py
----------------
Manages the tourist attraction graph using NetworkX.

A graph is a collection of:
  - Nodes  : tourist locations (e.g. "Eiffel Tower")
  - Edges  : travel paths between two locations
  - Weights: distance (km) stored on each edge

NetworkX stores everything in memory, so every time the Streamlit
app reruns we rebuild the graph from the lists kept in
st.session_state.  That is why every method works on plain Python
data structures (lists / dicts) that can be serialised by Streamlit,
and we only create the nx.Graph object when we need it.
"""

import networkx as nx


# ---------------------------------------------------------------------------
# GraphManager class
# ---------------------------------------------------------------------------

class GraphManager:
    """
    Wraps a NetworkX undirected weighted graph.

    Attributes
    ----------
    graph : nx.Graph
        The underlying NetworkX graph object.
    """

    def __init__(self):
        """Create an empty undirected graph."""
        self.graph = nx.Graph()

    # ------------------------------------------------------------------
    # Node (location) operations
    # ------------------------------------------------------------------

    def add_location(self, name: str) -> tuple[bool, str]:
        """
        Add a tourist location (node) to the graph.

        Parameters
        ----------
        name : str
            Name of the tourist location.

        Returns
        -------
        (success: bool, message: str)
        """
        name = name.strip()
        if not name:
            return False, "ชื่อสถานที่ไม่สามารถว่างได้"
        if self.graph.has_node(name):
            return False, f"'{name}' มีอยู่ในระบบแล้ว"
        self.graph.add_node(name)
        return True, f"เพิ่มสถานที่ '{name}' เรียบร้อยแล้ว"

    def remove_location(self, name: str) -> tuple[bool, str]:
        """
        Remove a tourist location (node) and all its connected paths.

        Parameters
        ----------
        name : str
            Name of the location to remove.

        Returns
        -------
        (success: bool, message: str)
        """
        if not self.graph.has_node(name):
            return False, f"ไม่พบ '{name}' ในระบบ"
        self.graph.remove_node(name)   # NetworkX also removes all edges attached to this node
        return True, f"ลบสถานที่ '{name}' และเส้นทางที่เชื่อมต่อทั้งหมดเรียบร้อยแล้ว"

    def get_locations(self) -> list[str]:
        """Return a list of all location names (node labels)."""
        return list(self.graph.nodes())

    # ------------------------------------------------------------------
    # Edge (path) operations
    # ------------------------------------------------------------------

    def add_path(self, loc1: str, loc2: str, distance: float) -> tuple[bool, str]:
        """
        Add a travel path (edge) between two locations with a distance weight.

        Parameters
        ----------
        loc1, loc2 : str
            Names of the two locations to connect.
        distance : float
            Distance in kilometres (must be > 0).

        Returns
        -------
        (success: bool, message: str)
        """
        if loc1 == loc2:
            return False, "เส้นทางต้องเชื่อมสถานที่สองแห่งที่แตกต่างกัน"
        if not self.graph.has_node(loc1):
            return False, f"ไม่พบสถานที่ '{loc1}' กรุณาเพิ่มสถานที่ก่อน"
        if not self.graph.has_node(loc2):
            return False, f"ไม่พบสถานที่ '{loc2}' กรุณาเพิ่มสถานที่ก่อน"
        if distance <= 0:
            return False, "ระยะทางต้องเป็นตัวเลขที่มากกว่า 0"
        if self.graph.has_edge(loc1, loc2):
            return False, f"มีเส้นทางระหว่าง '{loc1}' และ '{loc2}' อยู่แล้ว"

        self.graph.add_edge(loc1, loc2, weight=distance)
        return True, f"เพิ่มเส้นทาง '{loc1}' ↔ '{loc2}' ({distance} กม.) เรียบร้อยแล้ว"

    def update_path(self, loc1: str, loc2: str, new_distance: float) -> tuple[bool, str]:
        """
        Update the distance weight of an existing travel path.

        Parameters
        ----------
        loc1, loc2   : str   — Names of the connected locations.
        new_distance : float — New distance in kilometres (must be > 0).

        Returns
        -------
        (success: bool, message: str)
        """
        if not self.graph.has_edge(loc1, loc2):
            return False, f"ไม่มีเส้นทางระหว่าง '{loc1}' และ '{loc2}'"
        if new_distance <= 0:
            return False, "ระยะทางต้องเป็นตัวเลขที่มากกว่า 0"
        self.graph[loc1][loc2]["weight"] = new_distance
        return True, f"อัปเดตระยะทาง '{loc1}' ↔ '{loc2}' เป็น {new_distance} กม. เรียบร้อยแล้ว"

    def remove_path(self, loc1: str, loc2: str) -> tuple[bool, str]:
        """
        Remove the travel path between two locations.

        Parameters
        ----------
        loc1, loc2 : str
            Names of the connected locations.

        Returns
        -------
        (success: bool, message: str)
        """
        if not self.graph.has_edge(loc1, loc2):
            return False, f"ไม่มีเส้นทางระหว่าง '{loc1}' และ '{loc2}'"
        self.graph.remove_edge(loc1, loc2)
        return True, f"ลบเส้นทาง '{loc1}' ↔ '{loc2}' เรียบร้อยแล้ว"

    def get_paths(self) -> list[dict]:
        """
        Return all edges as a list of dicts.

        Each dict has keys: 'from', 'to', 'distance'.
        """
        paths = []
        for u, v, data in self.graph.edges(data=True):
            paths.append({
                "from": u,
                "to": v,
                "distance": data.get("weight", 0)
            })
        return paths

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def has_location(self, name: str) -> bool:
        """Return True if the location exists in the graph."""
        return self.graph.has_node(name)

    def has_path(self, loc1: str, loc2: str) -> bool:
        """Return True if a direct path exists between the two locations."""
        return self.graph.has_edge(loc1, loc2)

    def get_distance(self, loc1: str, loc2: str) -> float | None:
        """
        Return the distance (weight) of the direct path between two locations.
        Returns None if no edge exists.
        """
        if self.graph.has_edge(loc1, loc2):
            return self.graph[loc1][loc2]["weight"]
        return None

    def is_connected_subgraph(self, nodes: list[str]) -> bool:
        """
        Check whether the subgraph induced by the given nodes is connected,
        i.e. every selected location can be reached from every other one.
        """
        subgraph = self.graph.subgraph(nodes)
        return nx.is_connected(subgraph)

    # ------------------------------------------------------------------
    # Serialisation helpers (for Streamlit session_state persistence)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the graph into a plain dict so it can be stored in
        Streamlit session_state across reruns.

        Returns
        -------
        {
            "nodes": [str, ...],
            "edges": [{"from": str, "to": str, "distance": float}, ...]
        }
        """
        return {
            "nodes": self.get_locations(),
            "edges": self.get_paths(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphManager":
        """
        Reconstruct a GraphManager from the dict produced by to_dict().

        Parameters
        ----------
        data : dict
            Previously serialised graph data.

        Returns
        -------
        GraphManager
        """
        gm = cls()
        for node in data.get("nodes", []):
            gm.graph.add_node(node)
        for edge in data.get("edges", []):
            gm.graph.add_edge(edge["from"], edge["to"], weight=edge["distance"])
        return gm
