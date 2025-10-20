import struct

M = 4 # DEFAULT MAX CHILDREN PER NODE
m = M // 2 # MINIMUM CHILDREN PER NODE

# record: id, x, y

class RTreeNode:
    def __init__(self, node_id, is_leaf=False):
        self.is_leaf = is_leaf # True if leaf node, False if internal node
        self.node_id = node_id # Unique identifier for the node
        self.size = 0  # Current number of children
        self.children = [] # List of child nodes pointers (Rectangles (minx,miny,maxx,maxy)) or entries (Point data (x,x,y,y)) 
        self.bbox = (float('inf'), float('inf'), float('-inf'), float('-inf'))  # (minx, miny, maxx, maxy)

    def is_leaf_node(self):
        """Check if the node is a leaf node."""
        return self.is_leaf

    def min_xy(self): 
        """Return the minimum point(x,y) = minx, miny.""" # (left-bottom corner)
        return (self.bbox[0], self.bbox[1])
    
    def max_xy(self): 
        """Return the maximum point(x,y) = maxx, maxy.""" # (right-top corner)
        return (self.bbox[2], self.bbox[3])

    def mindist_to_point(self, point):
        """Calculate the minimum distance from the bounding box to a point."""
        px, py = point
        minx, miny, maxx, maxy = self.bbox
        if px < minx:
            dx = minx - px
        elif px > maxx:
            dx = px - maxx
        else:
            dx = 0
        if py < miny:
            dy = miny - py
        elif py > maxy:
            dy = py - maxy
        else:
            dy = 0
        return (dx * dx + dy * dy) ** 0.5 # Euclidean distance to the nearest edge or corner

    def update_bbox(self):
        """Update the bounding box to enclose all children."""
        if self.is_leaf:
            if not self.children:
                self.bbox = (float('inf'), float('inf'), float('-inf'), float('-inf'))
                return
            minx = min(child[0] for child in self.children)
            miny = min(child[1] for child in self.children)
            maxx = max(child[2] for child in self.children)
            maxy = max(child[3] for child in self.children)
            self.bbox = (minx, miny, maxx, maxy)
        else:
            # Primero actualizar bbox de todos los hijos
            for child in self.children:
                child.update_bbox()
            if not self.children:
                self.bbox = (float('inf'), float('inf'), float('-inf'), float('-inf'))
                return
            # Luego calcular bbox que englobe todos los hijos
            minx = min(child.bbox[0] for child in self.children)
            miny = min(child.bbox[1] for child in self.children)
            maxx = max(child.bbox[2] for child in self.children)
            maxy = max(child.bbox[3] for child in self.children)
            self.bbox = (minx, miny, maxx, maxy)
        
    def area(self):
        """Calculate the area of the bounding box."""
        minx, miny, maxx, maxy = self.bbox
        return (maxx - minx) * (maxy - miny)
    
    def enlarged_area(self, rect):
        """Calculate the area increase if this node's bbox were to include rect."""
        minx, miny, maxx, maxy = self.bbox
        rminx, rminy, rmaxx, rmaxy = rect
        new_minx = min(minx, rminx)
        new_miny = min(miny, rminy)
        new_maxx = max(maxx, rmaxx)
        new_maxy = max(maxy, rmaxy)
        return (new_maxx - new_minx) * (new_maxy - new_miny) - self.area()
    
class RTree:
    def __init__(self, max_children=M):
        self.root = RTreeNode(0)
        self.root.is_leaf = True
        self.max_children = max_children
        self.node_count = 1  # To assign unique IDs to nodes

    def is_empty(self):
        return self.root.is_leaf and self.root.size == 0

    def insert(self, record):
        """Insert a record given the record we want to add.

        Supported formats:
        - point record: (id, x, y)  -> stored as degenerate rect (x,y,x,y)
        - leaf entry (legacy): (minx, miny, maxx, maxy, id)
        """
        
        # Normalize incoming record to (rect_tuple, payload)
        if isinstance(record, tuple) and len(record) == 3:
            # (id, x, y)
            payload, x, y = record
            rect = (x, y, x, y)
        elif isinstance(record, tuple) and len(record) == 5:
            # legacy leaf entry (minx, miny, maxx, maxy, id)
            rect = (record[0], record[1], record[2], record[3])
            payload = record[4]
        else:
            raise ValueError("insert expects (id,x,y) or (minx,miny,maxx,maxy,id)")

        if self.root.is_leaf and self.root.size == 0:
            self.root.children.append((rect[0], rect[1], rect[2], rect[3], payload))
            self.root.size = 1
            self.root.update_bbox()
        else:
            self.insert_by_rect(rect, payload)

    def insert_by_rect(self, rect, record):
        """Insert a record into the R-Tree based on its bounding rectangle."""
        leaf = self._choose_leaf(self.root, rect)
        # store as leaf entry: (minx, miny, maxx, maxy, payload)
        leaf.children.append((rect[0], rect[1], rect[2], rect[3], record))
        leaf.size += 1
        leaf.update_bbox()
        if leaf.size > self.max_children:
            self._split_node(leaf)
    
    def _choose_leaf(self, node, rect):
        """Choose the appropriate leaf node for insertion."""
        if node.is_leaf:
            return node
        else:
            rect_center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
            def score(child):
                #child is RTreeNode
                return (child.enlarged_area(rect), child.mindist_to_point(rect_center))
            best_child = min(node.children, key = score) # Choose child that requires least enlargement and is closest
            return self._choose_leaf(best_child, rect)
    
    def _split_node(self, node):
        """Split a node that has exceeded max_children."""
        k = node.size
        # mínimo de entradas por nodo (ceil(M/2))
        min_fill = max(1, (self.max_children + 1) // 2)
        # elegir índice de corte garantizando que ambos lados tengan >= min_fill
        mid = max(min_fill, min(k // 2, k - min_fill))
        new_node = RTreeNode(self.node_count)
        self.node_count += 1
        new_node.is_leaf = node.is_leaf
        new_node.children = node.children[mid:]
        node.children = node.children[:mid]

        node.size = len(node.children)
        new_node.size = len(new_node.children)
        
        node.update_bbox()
        new_node.update_bbox()
        
        if node == self.root:
            new_root = RTreeNode(self.node_count)
            self.node_count += 1
            new_root.is_leaf = False
            new_root.children = [node, new_node]
            new_root.size = 2
            new_root.update_bbox()
            self.root = new_root
        else:
            parent = self._find_parent(self.root, node)
            parent.children.append(new_node)
            parent.size += 1
            parent.update_bbox()
            if parent.size > self.max_children:
                self._split_node(parent)

    def _find_parent(self, current, child):
        if current.is_leaf:
            return None
        for c in current.children:
            if c == child:
                return current
            if not c.is_leaf:  # Solo buscar recursivamente en nodos internos
                res = self._find_parent(c, child)
                if res:
                    return res
        return None

    def search(self, key):
        """Search for all records that intersect the given bounding box."""
        results = []
        self._search_recursive(self.root, key, results)
        return results
    
    def _search_recursive(self, node, key, results):
        if node.is_leaf:
            for child in node.children:
                if not (child[2] < key[0] or child[0] > key[2] or
                        child[3] < key[1] or child[1] > key[3]):
                    results.append(child[4])
        else:
            for child in node.children:
                if not (child.bbox[2] < key[0] or child.bbox[0] > key[2] or
                        child.bbox[3] < key[1] or child.bbox[1] > key[3]):
                    self._search_recursive(child, key, results)
    
    def rangeSearch(self, point, radius_or_k):
        """
        Range search with automatic mode detection:
        - If radius_or_k is a float: rangeSearch(point, radius) - search within circular area
        - If radius_or_k is an int: rangeSearch(point, k) - find k nearest neighbors
        """
        if isinstance(radius_or_k, float):
            return self._range_search_radius(point, radius_or_k)
        elif isinstance(radius_or_k, int):
            return self._range_search_k(point, radius_or_k)
        else:
            raise ValueError("Second parameter must be float (radius) or int (k)")
    
    def _range_search_radius(self, point, radius):
        """Range search within a circular area using bbox mindist pruning."""
        results = []
        def rect_tuple_mindist(rect, point):
            px, py = point
            minx, miny, maxx, maxy = rect[0], rect[1], rect[2], rect[3]
            dx = 0 if minx <= px <= maxx else min(abs(px - minx), abs(px - maxx))
            dy = 0 if miny <= py <= maxy else min(abs(py - miny), abs(py - maxy))
            return (dx * dx + dy * dy) ** 0.5
        def recurse(node):
            if node.mindist_to_point(point) > radius:
                return 
            if node.is_leaf:
                for child in node.children:
                    if rect_tuple_mindist(child, point) <= radius:
                        results.append(child[4])
            else:
                for child in node.children:
                    if child.mindist_to_point(point) <= radius:
                        recurse(child)
        recurse(self.root)
        return results

    def _range_search_k(self, point, k):
        """Range search for the k nearest neighbors to a point."""
        results = []
        def recurse(node):
            if node.is_leaf:
                for child in node.children:
                    cx = (child[0] + child[2]) / 2.0
                    cy = (child[1] + child[3]) / 2.0
                    dist = ((cx - point[0]) ** 2 + (cy - point[1]) ** 2) ** 0.5
                    results.append((dist, child[4]))
                results.sort(key=lambda x: x[0])
            else:
                mindists = [(child.mindist_to_point(point), child) for child in node.children]
                mindists.sort(key=lambda x: x[0])
                for _, child in mindists:
                    recurse(child)
                    if len(results) >= k:
                        break
        recurse(self.root)
        return [r for _, r in results[:k]]

    def intersection_search(self, bbox):
        """Search for all records intersecting the given bounding box."""
        results = []
        self._intersection_search_recursive(self.root, bbox, results)
        return results
    
    def _intersection_search_recursive(self, node, bbox, results):
        if node.is_leaf:
            for child in node.children:
                if not (child[2] < bbox[0] or child[0] > bbox[2] or
                        child[3] < bbox[1] or child[1] > bbox[3]):
                    results.append(child[4])
        else:
            for child in node.children:
                if not (child.bbox[2] < bbox[0] or child.bbox[0] > bbox[2] or
                        child.bbox[3] < bbox[1] or child.bbox[1] > bbox[3]):
                    self._intersection_search_recursive(child, bbox, results)

    def delete(self, key):
        """Delete a record from the R-Tree"""
        deleted_nodes = []
        self._delete_recursive(self.root, key, deleted_nodes)
        
        # Reinsert orphaned entries from deleted nodes
        for orphaned_entries in deleted_nodes:
            for entry in orphaned_entries:
                if isinstance(entry, RTreeNode):
                    # Reinsert subtree
                    self._reinsert_subtree(entry)
                else:
                    # Leaf entry: (minx, miny, maxx, maxy, payload)
                    minx, miny, maxx, maxy, payload = entry
                    # If degenerate (point) reinsert via insert(record)
                    if minx == maxx and miny == maxy:
                        self.insert((payload, minx, miny))
                    else:
                        # reinserción como rect original
                        self.insert_by_rect((minx, miny, maxx, maxy), payload)
 
        # If root has only one child and is not a leaf, make child the new root
        if not self.root.is_leaf and self.root.size == 1:
             self.root = self.root.children[0]
    
    def _delete_recursive(self, node, key, deleted_nodes):
        """Recursively search and delete the record"""
        if node.is_leaf:
            # Remove matching records from leaf node
            original_count = node.size
            node.children = [child for child in node.children if child[4] != key]
            node.size = len(node.children)
            if node.size < original_count:
                node.update_bbox()
                # Check for underflow (less than m entries where m = max_children/2)
                min_entries = max(1, self.max_children // 2)
                if node.size < min_entries and node != self.root:
                    # Node underflows - it will be deleted and entries reinserted
                    deleted_nodes.append(node.children[:])  # Save entries for reinsertion
                    node.children = []  # Mark node as deleted
                    node.size = 0
                    return True
            return len(node.children) < original_count
        else:
            # Internal node - search in children
            nodes_to_remove = []
            for i, child in enumerate(node.children):
                if self._delete_recursive(child, key, deleted_nodes):
                    # Child was modified or deleted
                    if not child.children:  # Child node was deleted (empty)
                        nodes_to_remove.append(i)
            
            # Remove deleted child nodes
            for i in reversed(nodes_to_remove):
                node.children.pop(i)
                node.size -= 1
            
            if node.size:  # If node still has children
                node.update_bbox()
                # Check for underflow in internal node
                min_entries = max(1, self.max_children // 2)
                if node.size < min_entries and node != self.root:
                    # Save all child subtrees for reinsertion
                    deleted_nodes.append(node.children[:])
                    node.children = []  # Mark node as deleted
                    node.size = 0
                    return True
            else:
                # Node has no children left
                return True
            
            return len(nodes_to_remove) > 0
    
    def _reinsert_subtree(self, subtree_root):
        """Reinsert a subtree that was orphaned during deletion"""
        if subtree_root.is_leaf_node():
            # Reinsert all entries in this leaf
            for entry in subtree_root.children:
                minx, miny, maxx, maxy, payload = entry
                if minx == maxx and miny == maxy:
                    self.insert((payload, minx, miny))
                else:
                    self.insert_by_rect((minx, miny, maxx, maxy), payload)
        else:
            # Recursively reinsert all subtrees
            for child in subtree_root.children:
                self._reinsert_subtree(child)

class RTreeIndex:
    def __init__(self, index_filename, fields, max_children=M, file_manager=None):
        self.index_filename = index_filename
        self.fields = fields  # List of field definitions for spatial data
        self.rtree = RTree(max_children=max_children)
        self.id_to_pos = {}
        self._loaded = False
        self.file_manager = file_manager

    def insert(self, record, pos=None):
        """Insert a record into the R-Tree index."""
        # soporte flexible para record (dict-like o objeto Record)
        def _get(rec, name):
            try:
                if isinstance(rec, dict):
                    return rec[name]
            except Exception:
                pass
            if hasattr(rec, name):
                return getattr(rec, name)
            try:
                return rec[name]
            except Exception:
                raise KeyError(name)

        # obtener coordenadas e id (intentar 'id' o 'key')
        rec_id = None
        try:
            rec_id = _get(record, 'id')
        except Exception:
            rec_id = getattr(record, 'key', None)
        x = _get(record, self.fields[0].name)
        y = _get(record, self.fields[1].name)
        # insertar en el RTree (usa la API de puntos (id,x,y))
        self.rtree.insert((rec_id, x, y))
        # guardar mapeo id -> posición en archivo si se suministró
        if pos is not None:
            self.id_to_pos[rec_id] = pos
        return rec_id

    def search(self, rec_id):
        """Return the file position for rec_id (or None)."""
        return self.id_to_pos.get(rec_id)
    
    def load_from_file(self):
        """Build the R-tree from the FileManager contents (if provided)."""
        if not self.file_manager:
            return False
        idx = 0
        any_inserted = False
        while True:
            record = self.file_manager.read_record(idx)
            if record is None:
                break
            # Only valid records (not logically deleted) - using .next == 0 convention
            if getattr(record, 'next', 0) == 0:
                try:
                    rec_id = getattr(record, 'key', None) or getattr(record, 'id', None)
                    x = getattr(record, self.fields[0].name)
                    y = getattr(record, self.fields[1].name)
                except Exception:
                    idx += 1
                    continue
                self.rtree.insert((rec_id, float(x), float(y)))
                self.id_to_pos[rec_id] = idx
                any_inserted = True
            idx += 1
        self._loaded = True
        return any_inserted

    def save_to_file(self):
        """No-op placeholder for persistence (implement if needed)."""
        return True

    # Optional wrappers for spatial queries returning ids
    def range_search(self, bbox):
        return self.rtree.intersection_search(bbox)

    def spatial_search(self, point, radius_or_k):
        """
        Búsqueda espacial unificada:
        - Si radius_or_k es float: búsqueda por radio
        - Si radius_or_k es int: k-vecinos más cercanos
        """
        return self.rtree.rangeSearch(point, radius_or_k)

    def bbox_search(self, bbox):
        """Búsqueda por rectángulo delimitador"""
        return self.rtree.intersection_search(bbox)

