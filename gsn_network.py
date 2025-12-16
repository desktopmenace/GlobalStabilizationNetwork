"""
GSN Network Analysis Module

Treats the Global Stabilization Network as a graph and analyzes
network properties including centrality, flow, and connectivity.

Uses NetworkX for graph analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations

# Try to import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[WARN] networkx not installed. Install with: pip install networkx")

# Try to import scipy for Delaunay triangulation
try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in km between two points."""
    R = 6371  # Earth's radius in km
    
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2


class GSNNetwork:
    """
    Represents the Global Stabilization Network as a graph.
    
    Nodes: Known sacred sites
    Edges: Connections based on distance, alignment, or energy flow
    """
    
    def __init__(self, nodes: Dict = None):
        """
        Initialize the network.
        
        Args:
            nodes: Dict mapping name -> (lat, lon) or name -> {"coords": (lat, lon), ...}
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for network analysis")
        
        self.G = nx.Graph()
        self._node_coords = {}
        
        if nodes:
            self._build_network(nodes)
    
    def _build_network(self, nodes: Dict):
        """Build network with multiple edge types."""
        # Parse nodes - handle both formats
        for name, data in nodes.items():
            if isinstance(data, tuple):
                lat, lon = data
            elif isinstance(data, dict):
                lat, lon = data.get("coords", (0, 0))
            else:
                continue
            
            self.G.add_node(name, lat=lat, lon=lon)
            self._node_coords[name] = (lat, lon)
        
        # Add edges of different types
        self._add_distance_edges(max_distance_km=5000)
        self._add_delaunay_edges()
        self._add_golden_edges()
    
    def _add_distance_edges(self, max_distance_km: float = 5000):
        """Add edges between nodes within a maximum distance."""
        nodes = list(self.G.nodes())
        
        for i, node1 in enumerate(nodes):
            lat1, lon1 = self._node_coords[node1]
            for node2 in nodes[i+1:]:
                lat2, lon2 = self._node_coords[node2]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                
                if dist <= max_distance_km:
                    self.G.add_edge(node1, node2, 
                                   type="distance", 
                                   distance=dist,
                                   weight=1.0 / (1 + dist/1000))
    
    def _add_delaunay_edges(self):
        """Add edges based on Delaunay triangulation (natural neighbors)."""
        if not HAS_SCIPY or len(self._node_coords) < 4:
            return
        
        nodes = list(self._node_coords.keys())
        coords = np.array([self._node_coords[n] for n in nodes])
        
        try:
            tri = Delaunay(coords)
            
            for simplex in tri.simplices:
                for i in range(3):
                    n1, n2 = nodes[simplex[i]], nodes[simplex[(i+1) % 3]]
                    if not self.G.has_edge(n1, n2):
                        lat1, lon1 = self._node_coords[n1]
                        lat2, lon2 = self._node_coords[n2]
                        dist = haversine_distance(lat1, lon1, lat2, lon2)
                        self.G.add_edge(n1, n2, 
                                       type="delaunay",
                                       distance=dist,
                                       weight=1.0)
        except Exception:
            pass  # Delaunay may fail for some configurations
    
    def _add_golden_edges(self, tolerance: float = 0.05):
        """Add edges between nodes at golden ratio distances."""
        base_distances = [100, 500, 1000, 2000]  # km
        golden_distances = []
        
        for base in base_distances:
            golden_distances.append(base * PHI)
            golden_distances.append(base * PHI * PHI)
        
        nodes = list(self.G.nodes())
        
        for i, node1 in enumerate(nodes):
            lat1, lon1 = self._node_coords[node1]
            for node2 in nodes[i+1:]:
                lat2, lon2 = self._node_coords[node2]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                
                # Check if distance matches any golden ratio distance
                for golden_dist in golden_distances:
                    if abs(dist - golden_dist) / golden_dist < tolerance:
                        if not self.G.has_edge(node1, node2):
                            self.G.add_edge(node1, node2,
                                           type="golden",
                                           distance=dist,
                                           golden_ratio=dist / base_distances[0],
                                           weight=1.5)  # Higher weight for golden edges
                        break
    
    def compute_network_metrics(self) -> Dict[str, Dict]:
        """Compute centrality and flow metrics for all nodes."""
        if len(self.G.nodes()) == 0:
            return {}
        
        metrics = {}
        
        try:
            # Degree centrality - how connected is each node
            degree = nx.degree_centrality(self.G)
            
            # Betweenness centrality - how often node is on shortest paths
            betweenness = nx.betweenness_centrality(self.G)
            
            # Eigenvector centrality - connected to other important nodes
            try:
                eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0.5 for n in self.G.nodes()}
            
            # Clustering coefficient - local interconnectedness
            clustering = nx.clustering(self.G)
            
            for node in self.G.nodes():
                metrics[node] = {
                    "degree": degree.get(node, 0),
                    "betweenness": betweenness.get(node, 0),
                    "eigenvector": eigenvector.get(node, 0),
                    "clustering": clustering.get(node, 0),
                }
        except Exception as e:
            print(f"[WARN] Error computing network metrics: {e}")
        
        return metrics
    
    def identify_hubs(self, top_n: int = 5) -> List[str]:
        """Identify the most central hub nodes."""
        metrics = self.compute_network_metrics()
        
        if not metrics:
            return []
        
        # Score by combined centrality
        hub_scores = {}
        for node, m in metrics.items():
            hub_scores[node] = (
                0.3 * m["degree"] + 
                0.3 * m["betweenness"] + 
                0.4 * m["eigenvector"]
            )
        
        sorted_nodes = sorted(hub_scores.items(), key=lambda x: -x[1])
        return [n for n, _ in sorted_nodes[:top_n]]
    
    def _connect_temp_node(self, temp_node: str, max_connections: int = 5):
        """Connect a temporary node to its nearest neighbors."""
        temp_lat = self.G.nodes[temp_node]["lat"]
        temp_lon = self.G.nodes[temp_node]["lon"]
        
        # Find distances to all other nodes
        distances = []
        for node in self.G.nodes():
            if node == temp_node:
                continue
            lat = self.G.nodes[node]["lat"]
            lon = self.G.nodes[node]["lon"]
            dist = haversine_distance(temp_lat, temp_lon, lat, lon)
            distances.append((node, dist))
        
        # Connect to nearest neighbors
        distances.sort(key=lambda x: x[1])
        for node, dist in distances[:max_connections]:
            self.G.add_edge(temp_node, node, type="temp", distance=dist)
    
    def _compute_bridge_score(self, node: str) -> float:
        """
        Compute how much a node bridges different clusters.
        High score = connects otherwise disconnected regions.
        """
        if self.G.degree(node) < 2:
            return 0.0
        
        neighbors = list(self.G.neighbors(node))
        
        # Check connectivity between neighbors without this node
        temp_graph = self.G.copy()
        temp_graph.remove_node(node)
        
        # Count how many neighbor pairs become disconnected
        disconnected_pairs = 0
        total_pairs = 0
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                total_pairs += 1
                if not nx.has_path(temp_graph, n1, n2):
                    disconnected_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return disconnected_pairs / total_pairs
    
    def compute_N_score(self, lat: float, lon: float) -> float:
        """
        Compute Network score for a potential new node location.
        
        High N score = location would improve network properties:
        - Bridges gaps between clusters
        - Strengthens weak connections
        - Maintains geometric harmony
        """
        if len(self.G.nodes()) < 3:
            return 0.5
        
        # Temporarily add node
        temp_node = f"_temp_{lat:.4f}_{lon:.4f}"
        self.G.add_node(temp_node, lat=lat, lon=lon)
        self._node_coords[temp_node] = (lat, lon)
        
        try:
            # Add edges to nearby nodes
            self._connect_temp_node(temp_node)
            
            if self.G.degree(temp_node) == 0:
                return 0.0
            
            # Compute improvement metrics
            betweenness = nx.betweenness_centrality(self.G)
            new_betweenness = betweenness.get(temp_node, 0)
            
            new_clustering = nx.clustering(self.G, temp_node)
            
            # Check if it bridges clusters
            bridge_score = self._compute_bridge_score(temp_node)
            
            # Combine into N score
            # High betweenness = on many paths = good
            # Low clustering = bridges different groups = good
            # High bridge score = connects disconnected regions = good
            N = (0.4 * new_betweenness * 10 +  # Scale up betweenness
                 0.3 * (1 - new_clustering) + 
                 0.3 * bridge_score)
            
            return float(np.clip(N, 0, 1))
            
        finally:
            # Remove temp node
            self.G.remove_node(temp_node)
            del self._node_coords[temp_node]
    
    def compute_flow_potential(self, lat: float, lon: float) -> float:
        """
        Compute flow potential based on network position.
        
        Uses electrical circuit analogy:
        - Lower effective resistance to hub nodes = better flow
        """
        if len(self.G.nodes()) < 3:
            return 0.5
        
        # Find hub nodes
        hubs = self.identify_hubs(top_n=3)
        
        if not hubs:
            return 0.5
        
        # Compute average distance to hubs
        distances = []
        for hub in hubs:
            hub_lat = self.G.nodes[hub]["lat"]
            hub_lon = self.G.nodes[hub]["lon"]
            dist = haversine_distance(lat, lon, hub_lat, hub_lon)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Convert to flow score (closer = better flow)
        # Normalize by typical inter-node distance
        flow_score = 1.0 / (1.0 + avg_distance / 2000)
        
        return float(flow_score)
    
    def get_edges(self) -> List[Dict]:
        """Get all edges with their data for visualization."""
        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "from_coords": self._node_coords.get(u),
                "to_coords": self._node_coords.get(v),
                "type": data.get("type", "unknown"),
                "distance": data.get("distance", 0),
                "weight": data.get("weight", 1.0),
            })
        return edges
    
    def get_node_count(self) -> int:
        """Get number of nodes in the network."""
        return len(self.G.nodes())
    
    def get_edge_count(self) -> int:
        """Get number of edges in the network."""
        return len(self.G.edges())


# Cached network instance
_cached_network = None
_cached_nodes_hash = None


def get_network(nodes: Dict = None) -> Optional[GSNNetwork]:
    """Get or create cached network instance."""
    global _cached_network, _cached_nodes_hash
    
    if not HAS_NETWORKX:
        return None
    
    if nodes is None:
        return _cached_network
    
    # Check if nodes changed
    nodes_hash = hash(frozenset(str(v) for v in nodes.values()))
    
    if _cached_network is None or nodes_hash != _cached_nodes_hash:
        try:
            _cached_network = GSNNetwork(nodes)
            _cached_nodes_hash = nodes_hash
        except Exception as e:
            print(f"[WARN] Failed to create network: {e}")
            return None
    
    return _cached_network


def compute_N_score(lat: float, lon: float, nodes: Dict = None) -> float:
    """Compute network score for a location."""
    network = get_network(nodes)
    
    if network is None:
        return 0.0
    
    return network.compute_N_score(lat, lon)


def get_network_stats(nodes: Dict = None) -> Dict:
    """Get network statistics."""
    network = get_network(nodes)
    
    if network is None:
        return {"available": False}
    
    return {
        "available": True,
        "nodes": network.get_node_count(),
        "edges": network.get_edge_count(),
        "hubs": network.identify_hubs(top_n=5),
    }
