import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import datetime

from meshdata.mesh_color import get_distinct_rgb_color, get_layer_rgb_color
from meshdata.mesh_graph import UnionFind, edge_key, check_flip, triangle_area_3d
from meshdata.mesh_io import write_obj, write_ply, save_graph


class Vertex:  
    def __init__(self, x=0, y=0, z=0,  index=-1, discrete_bins=None):  
        if discrete_bins is not None:  
            self.x = min(int((x + 1) * discrete_bins / 2), discrete_bins - 1)  
            self.y = min(int((y + 1) * discrete_bins / 2), discrete_bins - 1)  
            self.z = min(int((z + 1) * discrete_bins / 2), discrete_bins - 1) 
            self.discrete_bins=discrete_bins
        else:  
            self.x = x  
            self.y = y  
            self.z = z  
        
        # basic connected component
        self.i = index  
        self.CC_id=None
        self.CC_novalid_mask=False
        self.decode_CC=None

        # pre process for nonmanifold
        self.repeat_mask=False
        self.pre_CC_id=None
        self.NM_vert=None
        self.neighbor_points=None
        self.pure_faces=None
        self.neighbor_faces=[]
        self.NM_process=False
        self.CC_of_triangle=None
        self.CC_of_triangle_reindex=None

        # for meto
        self.bfs_layer=-1 # 0,1,2,3,...,n-1,n (maximum distance)
        self.bfs_order=-1 # 0,1,2,3,....,vert number
        self.bfs_mask=False # already visited by BFS
        self.order_mask=False # already visited by BFS

        self.start_half_edges=[]
        self.end_half_edges=[]

        # set bound info, before BFS
        self.is_bound=False
        self.neighbors=[] # counter-block-wise sorting, from bound
        

        # set layer view, after BFS, except for layer 0
        self.layer_view_neighbor=[]
        self.layer_view_neighbor_origin=[]
        self.s_zero_ind=None
        self.e_zero_ind=None
        self.neg1_list=[] # index of origin -1 
        self.zero_cnt_ori=None
        
        self.next_layer_order=[]
        self.next_layer_order_mask=[]
        self.max_layer=None
        self.is_top=False

    # should be done before BFS, just setting neighbors
    def set_bound_info(self, neighborlimit=100):
        if len(self.start_half_edges)*len(self.end_half_edges) == 0:
            raise Exception('[E] this vertex is lonely')
        
        # prechecking, if bound
        next_he=self.start_half_edges[0]
        self.neighbors.append(next_he.e.i)
        loop_cnt=0
        while True:
            loop_cnt+=1
            if loop_cnt>neighborlimit:
                raise Exception(f'[E] vertex neighbor exceed limit {neighborlimit}')
            if next_he.p.o is not None:
                next_he=next_he.p.o
                # back to origin
                if next_he.e.i == self.start_half_edges[0].e.i:
                    # if loop to the start, still checking
                    if next_he.o.p.p.s.i == self.i and next_he.o.p.p.e.i == self.neighbors[-1]:
                        # triangle
                        pass
                    elif next_he.o.p.p.p.s.i == self.i and next_he.o.p.p.p.e.i == self.neighbors[-1]:
                        # using -1 to indicate the quad face
                        self.neighbors.append(-1)
                        pass
                    else:
                        raise Exception('[E] wrong for neighbor checking 1')
                    
                    break

                # checking triangle or quad
                if next_he.o.p.p.s.i == self.i and next_he.o.p.p.e.i == self.neighbors[-1]:
                    # triangle
                    self.neighbors.append(next_he.e.i)
                elif next_he.o.p.p.p.s.i == self.i and next_he.o.p.p.p.e.i == self.neighbors[-1]:
                    self.neighbors.append(-1)
                    self.neighbors.append(next_he.e.i)
                else:
                    raise Exception('[E] wrong for neighbor checking 2')
                    
            else:
                # find bound
                self.is_bound=True
                break
        
        # if vert is bound
        if self.is_bound: # re checking if meeting bound
            start_he_list=[]
            self.neighbors=[]
            for he in self.start_half_edges:
                if he.o is None:
                    start_he_list.append(he)
            if len(start_he_list) != 1:
                raise Exception('[E] nonmanifold, pls checking preprocess')
            
            candidate_start_half_edge=[]
            candidate_end_half_edge=[]
            candidate_neighbors=[]

            for can_ind, can_he in enumerate(start_he_list):
                # self.bound_start_half_edge=start_he
                candidate_start_half_edge.append(can_he)
                this_neighbors=[]
                this_neighbors.append(can_he.e.i)
                loop_cnt=0
                while True:
                    loop_cnt+=1
                    if loop_cnt>neighborlimit:
                        raise Exception(f'[E] bound vertex neighbor exceed limit {neighborlimit}')
                    if can_he.p.o is not None:
                        can_he=can_he.p.o
                        # checking quad or triangle
                        if can_he.o.p.p.s.i == self.i and can_he.o.p.p.e.i == this_neighbors[-1]:
                            # triangle
                            this_neighbors.append(can_he.e.i)
                        elif can_he.o.p.p.p.s.i == self.i and can_he.o.p.p.p.e.i == this_neighbors[-1]:
                            # quad
                            this_neighbors.append(-1)
                            this_neighbors.append(can_he.e.i)
                        else:
                            raise Exception('[E] bound is not triangle or square')

                    else:
                        # find the end half edge
                        if can_he.n.n.e.i == self.i:
                            # triangle
                            this_neighbors.append(can_he.p.s.i)
                        elif can_he.n.n.n.e.i == self.i:
                            # quad
                            this_neighbors.append(-1)
                            this_neighbors.append(can_he.p.s.i)
                        else:
                            raise Exception('[E] bound last is not triangle or square')
                        candidate_end_half_edge.append(can_he.p)
                        break
                candidate_neighbors.append(this_neighbors)
            
            if len(candidate_neighbors)==1:
                self.neighbors=candidate_neighbors[0]
            else:
                choose_index=0
                cur_neighbor_num=len(candidate_neighbors[0])
                for ind_n in range(1, len(candidate_neighbors)):
                    if len(candidate_neighbors[ind_n]) > cur_neighbor_num:
                        cur_neighbor_num=len(candidate_neighbors[ind_n])
                        choose_index=ind_n
                self.neighbors=candidate_neighbors[choose_index]


    def undiscrete_vertex(self, color='cc_face'):
        discrete_bins=self.discrete_bins
        if color == 'none':
            return [  
                (self.x + 0.5) / discrete_bins * 2 - 1,  
                (self.y + 0.5) / discrete_bins * 2 - 1,  
                (self.z + 0.5) / discrete_bins * 2 - 1,]
        if color == 'cc_decode':
            cc_id=self.decode_CC # 1, 2, ...
            r, g, b = get_distinct_rgb_color(cc_id)
            return [  
                (self.x + 0.5) / discrete_bins * 2 - 1,  
                (self.y + 0.5) / discrete_bins * 2 - 1,  
                (self.z + 0.5) / discrete_bins * 2 - 1,
                r / 255,
                g / 255,
                b / 255]
        if color == 'layer':
            layer_id=self.bfs_layer # 0, 1, 2, ...
            r, g, b = get_layer_rgb_color(layer_id)
            return [  
                (self.x + 0.5) / discrete_bins * 2 - 1,  
                (self.y + 0.5) / discrete_bins * 2 - 1,  
                (self.z + 0.5) / discrete_bins * 2 - 1,
                r / 255,
                g / 255,
                b / 255]
        if color == 'cc_face':
            cc_id=self.CC_of_triangle_reindex # 1, 2, ...
            if cc_id is None:
                cc_id='B'

            r, g, b = get_distinct_rgb_color(cc_id)
            return [  
                (self.x + 0.5) / discrete_bins * 2 - 1,  
                (self.y + 0.5) / discrete_bins * 2 - 1,  
                (self.z + 0.5) / discrete_bins * 2 - 1,
                r / 255,
                g / 255,
                b / 255]

    
    def __add__(self, other):  
        if isinstance(other, Vertex):
            return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)  

    def __sub__(self, other):  
        if isinstance(other, Vertex):  
            return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)  

    def __eq__(self, other):  
        if isinstance(other, Vertex):  
            return (self.x == other.x and   
                    self.y == other.y and   
                    self.z == other.z)  

    def __lt__(self, other):  
        if isinstance(other, Vertex):  
            return (self.y < other.y or   
                    (self.y == other.y and self.z < other.z) or   
                    (self.y == other.y and self.z == other.z and self.x < other.x))
        

class Vector3f:  
    def __init__(self, x=0.0, y=0.0, z=0.0):  
        self.x = x  
        self.y = y  
        self.z = z  

    @classmethod  
    def from_vertex(cls, vertex):  
        return cls(vertex.x, vertex.y, vertex.z)  

    @classmethod  
    def from_vertices(cls, v1, v2):  
        return cls(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z)  

    def __add__(self, other):  
        if isinstance(other, Vector3f):  
            return Vector3f(self.x + other.x, self.y + other.y, self.z + other.z)  

    def __sub__(self, other):  
        if isinstance(other, Vector3f):  
            return Vector3f(self.x - other.x, self.y - other.y, self.z - other.z)  

    def __mul__(self, scalar):  
        return Vector3f(self.x * scalar, self.y * scalar, self.z * scalar)  

    def __truediv__(self, scalar):  
        return Vector3f(self.x / scalar, self.y / scalar, self.z / scalar)  

    def __eq__(self, other):  
        if isinstance(other, Vector3f):  
            return (self.x == other.x and   
                    self.y == other.y and   
                    self.z == other.z)  

    def __lt__(self, other):  
        if isinstance(other, Vector3f):  
            return (self.y < other.y or   
                    (self.y == other.y and self.z < other.z) or   
                    (self.y == other.y and self.z == other.z and self.x < other.x))  

    def cross(self, other):  
        if isinstance(other, Vector3f):  
            return Vector3f(  
                self.y * other.z - self.z * other.y,  
                self.z * other.x - self.x * other.z,  
                self.x * other.y - self.y * other.x  
            )  

    def dot(self, other):  
        if isinstance(other, Vector3f):  
            return self.x * other.x + self.y * other.y + self.z * other.z  

    def norm(self):  
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)  

    def normalize(self):  
        n = self.norm()  
        if n == 0:  
            raise ValueError("Cannot normalize a zero vector.")  
        return Vector3f(self.x / n, self.y / n, self.z / n)
    
class HalfEdge:  
    def __init__(self, v=None, s=None, e=None, t=None, n=None, p=None, o=None, index=-1):  
        self.s = s  # start vertex (Vertex class)  
        self.e = e  # end vertex (Vertex class)  
        self.t = t  # triangle (Facet class)  
        self.n = n  # next half edge (HalfEdge class)  
        self.p = p  # previous half edge (HalfEdge class)  
        self.o = o  # opposite half edge (HalfEdge class or None for bound)  
        self.i = index  # index  
        self.CC_id=None 

    @staticmethod  
    def _vertex_distance(v1, v2):  
        # distance of two vertex class
        if v1 is None or v2 is None:
            return float('inf')   
        return Vector3f(v1.x, v1.y, v1.z).norm() - Vector3f(v2.x, v2.y, v2.z).norm()
    
    def __lt__(self, other):  
        if isinstance(other, HalfEdge):  
            return (self.s.y < other.s.y or   
                    (self.s.y == other.s.y and self.s.z < other.s.z) or   
                    (self.s.y == other.s.y and self.s.z == other.s.z and self.s.x < other.s.x))


class Facet:  
    def __init__(self, vertices=None, half_edges=None, index=-1, component_index=-1, mark=0):  
        self.vertices = vertices if vertices is not None else [None] * 4  # store 4 vertices
        self.half_edges = half_edges if half_edges is not None else [None] * 4  # store 4 half-edges  
        self.i = index  # index  
        self.ic = component_index  # component index  
        self.m = mark  # visited mark  
        self.center = Vector3f()  # zero vector
        self.vert_FN = 0 
        self.fix_orientation=False
        self.cc_of_face = -1

    def mark_CC(self, cc_index):
        if self.cc_of_face == -1:
            self.cc_of_face = cc_index
        elif self.cc_of_face != cc_index:
            raise Exception('face mark cc wrong')

    def flip(self):  
        # equal to flipping half-edges
        for he in self.half_edges:  
            if he:  
                he.s, he.e = he.e, he.s  
                he.n, he.p = he.p, he.n

    def __lt__(self, other):  
        if not isinstance(other, Facet):  
            return NotImplemented  

        if self.ic != other.ic:  
            return self.ic < other.ic  
        return self.center < other.center  

class Mesh:  
    def __init__(self, vertices, triangles, discrete_bins=256, verbose=False, debugging=False, non_mani_process=True, NM_max_edge_graph=50, NM_max_nonmani_verts=500, min_CC_face=1, M2_path=None, just_process=False):  
        # Mesh data  
        self.verts = []  # Vertex class 
        self.faces = []  # Facet class
        self.verbose = verbose  # show more info
        self.discrete_bins = discrete_bins  # quantization resolution
        self.debugging=debugging # for debugging
        self.non_mani_process=non_mani_process # for saving time
        self.num_vertices_p=0 # vertices num after this process (may remove repeated/lonely vertices)
        self.just_process=just_process

        self.M2_path=M2_path
        # Filter too complex meshes
        self.NM_edge_graph_limit=NM_max_edge_graph
        self.NM_nonmani_vert_limit=NM_max_nonmani_verts
        # Filter too small CC
        self.min_CC_face=min_CC_face
        
        if self.debugging:
            self.verbose=True

        # [step 1] init verts, remove repeat verts due to quantization ########################
        repeat_map_dic={}
        for i, vertex in enumerate(vertices):  
            v = Vertex(vertex[0], vertex[1], vertex[2], i, discrete_bins)
            # check repeated verts
            for ind, ele in enumerate(self.verts):
                if ele == v:
                    self.num_vertices_p-=1
                    if len(self.verts) in repeat_map_dic.keys():
                        raise Exception('[E] repeat vert wrong')
                    repeat_map_dic[len(self.verts)]=ind  # repeated vertex index ---> index first appear
                    v.repeat_mask=True
                    break
            self.verts.append(v)
            self.num_vertices_p+=1

        self.num_vertices_all = len(self.verts) # may including repeated vertex
        self.merge_repeat_verts_num=len(repeat_map_dic.keys()) # repeated vertex num
        self.num_faces_all= len(triangles)

        if self.verbose:
            # print(f'check repeat map dic: {repeat_map_dic}')
            print(f'merge {self.merge_repeat_verts_num} repeated verts due to quantization')

        # [step 2] remove faces, including repeated and degraded ###################################
        if triangles.shape[1] != 3:
            raise Exception('only support triangles now')
        
        degraded_face_index1=[] # same vertex index
        degraded_face_index2=[] # 3 vertices on a line
        
        replace_face_dic={}
        
        for ind_f in range(len(triangles)):
            face_line=triangles[ind_f]
            new_face_line=[]
            replace_mark=False
            for ind_fv in range(len(face_line)):
                if face_line[ind_fv] not in repeat_map_dic.keys():
                    new_face_line.append(face_line[ind_fv])
                    continue
                else:
                    new_face_line.append(repeat_map_dic[face_line[ind_fv]])
                    replace_mark=True
            if replace_mark == True:
                replace_face_dic[ind_f]=new_face_line
                
            
            if len(set(new_face_line))!=3:
                degraded_face_index1.append(ind_f)
                continue

            v1, v2, v3 = self.verts[new_face_line[0]], self.verts[new_face_line[1]], self.verts[new_face_line[2]],
            area=triangle_area_3d(v1.undiscrete_vertex(color='none'), v2.undiscrete_vertex(color='none'), v3.undiscrete_vertex(color='none')) 
            if area < 1e-5:
                degraded_face_index2.append(ind_f)


        for key in replace_face_dic.keys():
            triangles[key]=replace_face_dic[key]

        self.replace_facevert_num=len(replace_face_dic.keys())
        
        if self.verbose:
            print(f'replace {self.replace_facevert_num} faces due to repeated vertex')

        degraded_face_index=degraded_face_index1+degraded_face_index2
        self.degraded_face_num=len(degraded_face_index)
        # mask degraded triangles
        mask_degraded = np.ones(triangles.shape[0], dtype=bool)  
        mask_degraded[degraded_face_index] = False  

        
        triangles=triangles[mask_degraded] # reduce triangles
        
        if self.verbose:
            print(f'remove {self.degraded_face_num} faces ({len(degraded_face_index1)}+{len(degraded_face_index2)}) due to degraded to edge/vertex')

        # move repeated face, including different orientation!!!
        self.move_repeat_face_num=len(triangles)

        unique_tri_sort=[]
        unique_tri=[]
        for f in triangles:
            f_sort=tuple(np.sort(f))
            if f_sort in unique_tri_sort:
                continue
            unique_tri_sort.append(f_sort)
            unique_tri.append(f)
        triangles=np.array(unique_tri)
        
        self.move_repeat_face_num=self.move_repeat_face_num-len(triangles)
        if self.verbose:
            print(f'remove {self.move_repeat_face_num} faces due to repeat')
        
        # [step 3] non-manifold preprocess ###################################

        if self.verbose:
            start_mani=time.time()

        self.non_manifold_vertex_cnt=0
        self.non_manifold=False
        self.non_manifold_born_vert_num=0

        if self.non_mani_process:
            # 1v2cc: nonmanifold vertex with two or more disconnected faces
            triangles, vertnum_1v2cc=self.pre_process_NM(triangles)
            if self.verbose:
                print(f'NM pre process, 1v2cc num: {vertnum_1v2cc}')

            triangles = self.first_process_NM(triangles)

            triangles, vertnum_1v2cc=self.post_process_NM(triangles)

            if self.verbose:
                print(f'NM post process, 1v2cc num: {vertnum_1v2cc}')
        
        if self.verbose:
            end_mani=time.time()
            print(f'non-mani preprocess time {end_mani-start_mani}s')
            
        
        triangle_cc_final=[c for c in self.triangle_CC_list]

        if -1 in triangle_cc_final:
            raise Exception('all face should find their CC')
        # coloring vertex
        for index, triangle in enumerate(triangles):
            cc_root=self.uf.find(self.triangle_CC_list[index])
            if not self.verts[triangle[0]].CC_of_triangle:
                self.verts[triangle[0]].CC_of_triangle=cc_root
            else:
                if self.uf.find(self.verts[triangle[0]].CC_of_triangle) != cc_root:
                    raise Exception('face-vert cc wrong')
                    
            if not self.verts[triangle[1]].CC_of_triangle:
                self.verts[triangle[1]].CC_of_triangle=cc_root
            else:
                if self.uf.find(self.verts[triangle[1]].CC_of_triangle) != cc_root:
                    raise Exception('face-vert cc wrong')
                    
            if not self.verts[triangle[2]].CC_of_triangle:
                self.verts[triangle[2]].CC_of_triangle=cc_root
            else:
                if self.uf.find(self.verts[triangle[2]].CC_of_triangle) != cc_root:
                    raise Exception('face-vert cc wrong')
            
        parent_list, _ = self.uf.parent_list()
        unique_roots = list(set(self.uf.find(i) for i in range(len(parent_list))))
        self.unique_roots=unique_roots
        # valid vertex: used by triangles and will be colored by CC
        valid_vertex_cnt=0 
        lonely_vertex_cnt=0
        for v in self.verts:
            if v.CC_of_triangle is not None:
                valid_vertex_cnt+=1
                v.CC_of_triangle_reindex=unique_roots.index(v.CC_of_triangle)+1 # 1,2,3,...
            else:
                lonely_vertex_cnt+=1

        if self.verbose:
            print(f'after nonmanifold pre processing, there are {valid_vertex_cnt}/{lonely_vertex_cnt}: valid/lonely vertices')
            mani_str='manifold'
            if self.non_manifold:
                mani_str='non-manifold'
            print(f'this mesh is {mani_str} mesh')
            # print(f'the connected component number is {len(unique_roots)} merged from {len(parent_list)}')

        if self.debugging or self.just_process:
            
            manidebug_verts=np.asarray([vert.undiscrete_vertex(color='cc_face') for vert in self.verts])
            write_ply(manidebug_verts, triangles, self.M2_path)
            
        # [step 4] half-edge for face contruction ###################################
         
        edge2halfedge = {}  # using dic to map edge to half-edge

        # CC
        self.CC_all_half_edges=[]
        self.CC_all_half_edges_in=[]
        self.CC_all_half_edges_bound=[]
        self.CC_faces=[]
        self.fliped_face_cnt=0
        self.CC_faces_group=[[] for i in range(len(unique_roots))]
       
       
        he_notright_cnt=0
        for i, triangle in enumerate(triangles):  
            f = Facet()  
            f.i = i  
            # triangle maybe 3 or 4
            if triangle[-1] == -1 or triangle.shape[0] == 3:
                FN = 3
            else:
                FN = 4

            f.vert_FN = FN
            if FN != 3:
                raise Exception('not support quad mesh now')
            
            self.CC_faces_group[self.verts[triangle[0]].CC_of_triangle_reindex-1].append(f.i)
            

            for j in range(FN):  
                e = HalfEdge()  
                e.s = self.verts[triangle[(j + 1) % FN]]  # start vert itself
                e.e = self.verts[triangle[(j + 2) % FN]]  # end vert itself
                e.t = f  # half edge's belonging face
                e.i = j  # half edge's index \in FN, 0/1/2/3
                f.vertices[j] = self.verts[triangle[j]]  
                f.half_edges[j] = e  

                f.mark_CC(f.vertices[j].CC_of_triangle_reindex)
                
                
                key = edge_key(triangle[(j + 1) % FN], triangle[(j + 2) % FN])  
                
                if key in edge2halfedge:  
                    
                    if edge2halfedge[key] is None:
                        
                        raise Exception(f'[E] if nonmainfold pre process: {self.non_mani_process}, meet non-manifold')

                    if edge2halfedge[key].s.i == e.e.i and edge2halfedge[key].e.i == e.s.i:
                         
                        e.o = edge2halfedge[key]  
                        edge2halfedge[key].o = e  
                        # mark done as None
                        edge2halfedge[key] = None
                    else:
                        # we neet to flip face
                        e.o = edge2halfedge[key]
                        edge2halfedge[key].o = e  
                        # mark done as None 
                        edge2halfedge[key] = None
                        he_notright_cnt+=1

                else:  
                    edge2halfedge[key] = e  
            
            # connect prev/next half-edge
            for j in range(FN):  
                f.half_edges[j].n = f.half_edges[(j + 1) % FN]  
                f.half_edges[j].p = f.half_edges[(j - 1) % FN]   #origin (j+2)

            # cal face center
            if FN ==3:
                f.center = Vector3f(  
                (f.vertices[0].x + f.vertices[1].x + f.vertices[2].x) / 3.0,  
                (f.vertices[0].y + f.vertices[1].y + f.vertices[2].y) / 3.0,  
                (f.vertices[0].z + f.vertices[1].z + f.vertices[2].z) / 3.0  
            )
            else:
                f.center = Vector3f(  
                (f.vertices[0].x + f.vertices[1].x + f.vertices[2].x + f.vertices[3].x) / 4.0,  
                (f.vertices[0].y + f.vertices[1].y + f.vertices[2].y + f.vertices[3].y) / 4.0,  
                (f.vertices[0].z + f.vertices[1].z + f.vertices[2].z + f.vertices[3].z) / 4.0  
            )  

            self.faces.append(f)

        if self.verbose:
            print(f'there are {he_notright_cnt} half-edges wrong, fixing')

        self.num_faces_p = len(self.faces)  
        
        
        # [step 5] flip check and repair ###################################
        # can not handle mobius cycle
        # process per faces group
        triangle_flip_index_list=[]
        self.not_success_flip_face=[]
        for group_index, cc_faces_group in enumerate(self.CC_faces_group):
            
            for i in cc_faces_group:
                f = self.faces[i]
                if f.fix_orientation:
                    continue
                
                f.fix_orientation=True
                next_neighbors_queue=[]
                FN = f.vert_FN
                if FN == 4:
                    raise Exception('not support square face now')
                ret, next_neighbors_queue, success = check_flip(f, FN)
                if ret:
                    triangle_flip_index_list.append(f.i)
                if not success:
                    self.not_success_flip_face.append(f.i)
                
                if next_neighbors_queue:
                    while True:
                        # print(next_neighbors_queue)
                        if not next_neighbors_queue:
                            break
                        
                        f_ready_index=next_neighbors_queue.pop(0)
                        f_ready=self.faces[f_ready_index]
                        if f_ready.fix_orientation==True:
                            continue
                        f_ready.fix_orientation=True
                        ret, next_neighbors_new, success = check_flip(f_ready, f_ready.vert_FN)
                        if ret:
                            triangle_flip_index_list.append(f_ready.i)
                        if next_neighbors_new:
                            next_neighbors_queue+=next_neighbors_new
                        if not success:
                            self.not_success_flip_face.append(f_ready.i)

        if self.verbose:
            print(f'face flipping done')
            print(f'not success flip face: {self.not_success_flip_face}')

        if len(self.not_success_flip_face)!=0:
            # can not fix faces flipping because of mobius cycle canused by mesh quantization
            if just_process:
                print('can not flipping fixing because of mobius cycle is caused')
                return
            else:
                raise Exception(f'mobius cycle topology: {len(self.not_success_flip_face)} faces flip fixing wrong')

        if len(set(triangle_flip_index_list))!=len(triangle_flip_index_list):
            raise Exception('[E] flip face repeat!')

        for index_flip in triangle_flip_index_list:
            a,b,c=triangles[index_flip][0],triangles[index_flip][1],triangles[index_flip][2]
            triangles[index_flip]=[a,c,b]
        

        self.fliped_face_cnt=len(triangle_flip_index_list)
        self.faces.sort(key=lambda f: f)
        if self.verbose:
            print(f'flip {self.fliped_face_cnt}/{len(triangles)} faces for fixing')
            
        if just_process:
            
            flipdebug_verts=np.asarray([vert.undiscrete_vertex(color='cc_face') for vert in self.verts])
            write_ply(flipdebug_verts, triangles, self.M2_path.replace('M2','M2f'))
            return

        # [step 6] CC recalculation ###################################
        
        self.num_components = 0  
        for i in range(len(triangles)):  
            f = self.faces[i]
            
            if f.ic == -1:  
                self.num_components += 1  
                
                queue = [f]  
                while queue:  
                    f = queue.pop(0)  
                    if f.ic != -1:  
                        continue  
                    # if face is not marked ic ---> ic==-1
                    f.ic = self.num_components
                    FN = f.vert_FN
                    for j in range(FN):  
                        e = f.half_edges[j]
                        if e.o is not None and e.o.t.ic == -1:  
                            queue.append(e.o.t)  

        if self.verbose:
            print(f'Connected component num {self.num_components}')

        for j in range(self.num_components):
            self.CC_all_half_edges.append([])
            self.CC_all_half_edges_in.append([])
            self.CC_all_half_edges_bound.append([])
            self.CC_faces.append([])

        # [step 7] classify half-edge based on cc ###################################
        exist_dic={}
        new_vert_cnt=0
        for i in range(len(triangles)):
            f=self.faces[i]
            comp_id=f.ic-1
            self.CC_faces[comp_id].append(f)
            FN = f.vert_FN
            for j in range(FN):   
                e=f.half_edges[j]

                if self.verts[e.s.i].CC_id is not None and self.verts[e.s.i].CC_id != comp_id:
                    copy_vert_i=e.s.i
                    if copy_vert_i in exist_dic:
                        if comp_id in exist_dic[copy_vert_i]:
                            vert_id=exist_dic[copy_vert_i][comp_id]
                            e.s=self.verts[vert_id]
                            e.p.e=self.verts[vert_id]
                            if e.o is not None:
                                e.o.e=self.verts[vert_id]
                            if e.p.o is not None:
                                e.p.o.s = self.verts[vert_id]
                    else:
                        new_index=self.num_vertices_all
                        copy_v = Vertex(self.verts[copy_vert_i].x, self.verts[copy_vert_i].y, self.verts[copy_vert_i].z, new_index)
                        copy_v.discrete_bins=self.discrete_bins
                        copy_v.CC_id=comp_id
                        self.verts.append(copy_v)
                        self.num_vertices_all+=1
                        new_vert_cnt+=1
                        # change half edge
                        e.s=copy_v
                        e.p.e=copy_v
                        if e.o is not None:
                            e.o.e = copy_v
                        if e.p.o is not None:
                            e.p.o.s = copy_v
                        if copy_vert_i in exist_dic:
                            exist_dic[copy_vert_i][comp_id]=new_index
                        else:
                            exist_dic[copy_vert_i]={}
                            exist_dic[copy_vert_i][comp_id]=new_index

                else:
                    self.verts[e.s.i].CC_id=comp_id
                
                if self.verts[e.e.i].CC_id is not None and self.verts[e.e.i].CC_id != comp_id:
                     
                    copy_vert_i=e.e.i
                    if copy_vert_i in exist_dic:
                        if comp_id in exist_dic[copy_vert_i]:
                            vert_id=exist_dic[copy_vert_i][comp_id]
                            e.e=self.verts[vert_id]
                            e.n.s=self.verts[vert_id]
                            if e.o is not None:
                                e.o.s=self.verts[vert_id]
                            if e.n.o is not None:
                                e.n.o.e = self.verts[vert_id]
                    else:
                        new_index=self.num_vertices_all
                        copy_v = Vertex(self.verts[copy_vert_i].x, self.verts[copy_vert_i].y, self.verts[copy_vert_i].z, new_index)
                        copy_v.discrete_bins=self.discrete_bins
                        copy_v.CC_id=comp_id
                        self.verts.append(copy_v)
                        self.num_vertices_all+=1
                        new_vert_cnt+=1
                        # change half edge
                        e.e=copy_v
                        e.n.s=copy_v
                        if e.o is not None:
                            e.o.s = copy_v
                        if e.n.o is not None:
                            e.n.o.e = copy_v
                        if copy_vert_i in exist_dic:
                            exist_dic[copy_vert_i][comp_id]=new_index
                        else:
                            exist_dic[copy_vert_i]={}
                            exist_dic[copy_vert_i][comp_id]=new_index
                
                else:
                    self.verts[e.e.i].CC_id=comp_id

                e.CC_id=comp_id

                # collect CC half edges
                if e.o is None:
                    self.CC_all_half_edges_bound[comp_id].append(e)
                else:
                    self.CC_all_half_edges_in[comp_id].append(e)

                self.CC_all_half_edges[comp_id].append(e)

        if self.verbose:
            print(f'Add {new_vert_cnt} verts due to 1v2cc')

        if new_vert_cnt !=0:
            raise Exception('pre process non-manifold wrong')
        
        self.new_generated_verts=new_vert_cnt
        self.num_vertices_p+=self.new_generated_verts
        # sort face
        self.faces.sort(key=lambda f: f)  
        for i in range(self.num_components):
            self.CC_faces[i].sort(key=lambda f: f)
        self.triangles_process=triangles

        # [step 8] CC mask ###################################
        # chcek small CC (< self.min_CC_face) and mask
        
        self.CC_mask=[]
        CC_invalid_verts=[]
        self.CC_invalid_face_cnt=0
        for CC_i in range(self.num_components):
            if len(self.CC_faces[CC_i])<self.min_CC_face:
                self.CC_mask.append(False)
                self.CC_invalid_face_cnt+=len(self.CC_faces[CC_i])
                for f in self.CC_faces[CC_i]:
                    for j in range(f.vert_FN):
                        CC_invalid_verts.append(f.vertices[j].i)
            else:
                self.CC_mask.append(True)
        self.num_components_valid=sum(self.CC_mask)
        self.CC_invalid_verts_cnt=len(set(CC_invalid_verts))
        self.num_vertices_p=self.num_vertices_p-self.CC_invalid_verts_cnt
        self.num_faces_p=self.num_faces_p-self.CC_invalid_face_cnt
        if self.verbose:
            print(f'CC valid: {self.num_components_valid}/{self.num_components}')
            print(f'CC invalid verts {self.CC_invalid_verts_cnt}')
            print(f'CC invalid faces {self.CC_invalid_face_cnt}')

    ##### for flip checking
    def bound_graph(self, unsuccess_face_index):
        graph=nx.Graph()
        bound_edges=[]
        flip_index_recheck=[]
        
        for f_i in unsuccess_face_index:
            temp_bound_edges=[]
            f=self.faces[f_i]
            for j in range(3):
                this_he=f.half_edges[j]
                if this_he.o is None:
                    continue
                if this_he.s.i == this_he.o.s.i and this_he.e.i == this_he.o.e.i:
                    temp_bound_edges.append(edge_key(this_he.s.i, this_he.e.i))
            if len(temp_bound_edges) == 3:
                flip_index_recheck.append(f_i)
                f.flip()
            else:
                bound_edges+=temp_bound_edges

        graph.add_edges_from(bound_edges)
        nodes_list = list(graph.nodes())
        for ele in nodes_list:
            self.verts[ele].CC_of_triangle_reindex = None

        save_graph(graph, self.M2_path.replace('M2','M2Graph').replace('.obj','.png'))

        return flip_index_recheck

        
                                       

    ##### pre process NM
    def NM_oneCC(self, vert_index, edge_to_face_dic):
        graph=nx.Graph()
        graph.add_edges_from(edge_to_face_dic.keys())
        components=list(nx.connected_components(graph))
        f_CCs=[]
        f_CCs_all=[]
        for component_nodes in components:  
            if len(component_nodes)<2:
                raise Exception('[NM] graph single node, why')
            CC_edge_list=[edge_key(edge[0], edge[1]) for edge in graph.edges(component_nodes)]
            f_CCs.append([edge_to_face_dic[edge] for edge in CC_edge_list])
            f_CCs_all+=[edge_to_face_dic[edge] for edge in CC_edge_list]
        if len(f_CCs_all) != len(edge_to_face_dic.keys()) or len(set(f_CCs_all)) != len(edge_to_face_dic.keys()):
            raise Exception('[NM] not equal, why')
        return f_CCs
    
    def NM_oneCC_post(self, vert_index, edge_to_face_dic):
        graph=nx.Graph()
        graph.add_edges_from(edge_to_face_dic.keys())
        components=list(nx.connected_components(graph))
        f_CCs=[]
        f_CCs_all=[]
        for component_nodes in components:  
            if len(component_nodes)<2:
                raise Exception('[NM] graph single node, why')
            
            CC_edge_list=[edge_key(edge[0], edge[1]) for edge in graph.edges(component_nodes)]
            f_CCs.append([edge_to_face_dic[edge] for edge in CC_edge_list])
            f_CCs_all+=[edge_to_face_dic[edge] for edge in CC_edge_list]
            if not nx.is_tree(graph.subgraph(component_nodes)):
                # must be cycle:
                if len(component_nodes)!=len(CC_edge_list):
                    raise Exception('[NM] post not manifold')
            else:
                # must be chain
                if len(component_nodes)!=len(CC_edge_list)+1:
                    raise Exception('[NM] post not manifold')
        if len(f_CCs_all) != len(edge_to_face_dic.keys()) or len(set(f_CCs_all)) != len(edge_to_face_dic.keys()):
            raise Exception('[NM] not equal, why')
        return f_CCs
    
    def pre_process_NM(self, triangles):
        face_group_per_vert=[{} for i in range(self.num_vertices_all)]
        vertnum_1v2cc=0
        face_vert_map_dic={}
        for face_index, face in enumerate(triangles):
            face_group_per_vert[face[0]][edge_key(face[1], face[2])]=face_index
            face_group_per_vert[face[1]][edge_key(face[2], face[0])]=face_index
            face_group_per_vert[face[2]][edge_key(face[0], face[1])]=face_index
        for vert_index, face_group in enumerate(face_group_per_vert):
            if not face_group:
                continue
            face_classify_list=self.NM_oneCC(vert_index, face_group)
            if len(face_classify_list) == 1:
                continue
            elif len(face_classify_list) ==0:
                raise Exception('face classify wrong')
            else:
                # 1v2cc situation
                self.non_manifold=True
                self.non_manifold_vertex_cnt+=1 # just 1v2cc monmani vertex
                vertnum_1v2cc+=1
                triangles, _=self.generate_new_vertex_for_NM(face_classify_list, vert_index, triangles)
                
        return triangles, vertnum_1v2cc
    
    def post_process_NM(self, triangles):
        face_group_per_vert=[{} for i in range(self.num_vertices_all)]
        vertnum_1v2cc=0
        face_vert_map_dic={}
        for face_index, face in enumerate(triangles):
            face_group_per_vert[face[0]][edge_key(face[1], face[2])]=face_index
            face_group_per_vert[face[1]][edge_key(face[2], face[0])]=face_index
            face_group_per_vert[face[2]][edge_key(face[0], face[1])]=face_index
        for vert_index, face_group in enumerate(face_group_per_vert):
            if not face_group:
                continue
            face_classify_list=self.NM_oneCC_post(vert_index, face_group)
            root_list=self.union_f_cc(face_classify_list)
            if len(face_classify_list) == 1:
                continue
            elif len(face_classify_list) ==0:
                raise Exception('face classify wrong')
            else:
                # 1v2cc situation
                self.non_manifold_vertex_cnt+=1 # just 1v2cc monmani vertex
                self.non_manifold=True
                vertnum_1v2cc+=1
                triangles, _=self.generate_new_vertex_for_NM(face_classify_list, vert_index, triangles)
            self.update_face_CC(face_classify_list, root_list)

        return triangles, vertnum_1v2cc
    
    def generate_new_vertex_for_NM(self, face_classify_list, vert_index, triangles):
        face_new_index_dic={}
        
        for f_classify_index in range(1, len(face_classify_list)):
            face_classified=face_classify_list[f_classify_index]
            # face of this_group: orivert ---> copy_x.i
            # copy vertex
            copy_vert_i = vert_index
            new_index=self.num_vertices_all
            copy_x = Vertex(self.verts[copy_vert_i].x, self.verts[copy_vert_i].y, self.verts[copy_vert_i].z, new_index)
            copy_x.discrete_bins=self.discrete_bins
            copy_x.neighbor_faces=face_classified
            self.verts.append(copy_x)
            self.num_vertices_all+=1
            self.num_vertices_p+=1
            self.non_manifold_born_vert_num+=1
            

            for face_i in face_classified:
                face_new_index_dic[face_i]=copy_x.i
        influenced_neighbor=[]
        temp_face_change_dic={}
        for face_i in face_new_index_dic.keys():
            target_index=face_new_index_dic[face_i]
            tria=triangles[face_i]
            influenced_neighbor+=[ele for ele in tria if ele!=vert_index]
            replace_dic={vert_index: target_index}
            new_tria=[replace_dic.get(item, item) for item in tria]
            temp_face_change_dic[face_i]=new_tria
        for key in temp_face_change_dic.keys():
            triangles[key]=temp_face_change_dic[key]

        self.verts[vert_index].neighbor_faces=face_classify_list[0]
        return triangles, list(set(influenced_neighbor))

    ### first process NM
    def find_cycle_or_chain(self, component):  
        # component: subgraph
        if not nx.is_tree(component):  
            # find all easy cycle 
            cycles = list(nx.simple_cycles(component.to_directed()))  
            if cycles:  
                max_cycle = max(cycles, key=len)  
                return max_cycle, 'cycle'  
                # max_cycle = min(cycles, key=len)  
                # return max_cycle, 'cycle'  
            raise Exception('can not find cycle ?')
    
        # longest chain
        longest_path = []  
        for node in component.nodes():  
            # dfs for chain 
            paths = nx.single_source_dijkstra_path(component, node)  
            for path in paths.values():  
                if len(path) > len(longest_path):  
                    longest_path = path  
        return longest_path, 'chain'


    def NM_init_edge_graph(self, vert_index, edge_to_face_dic):
        nonmani_vert=False
        graph=nx.Graph()
        graph.add_edges_from(edge_to_face_dic.keys())
        components=list(nx.connected_components(graph))
        if len(components)!=1:
            raise Exception('[NM] still meet 1v2cc situation, why')
        cycle_or_chain, type = self.find_cycle_or_chain(graph)
        if type == 'cycle':
            cc_edges=[edge_key(cycle_or_chain[i%len(cycle_or_chain)], cycle_or_chain[(i+1)%len(cycle_or_chain)]) for i in range(len(cycle_or_chain))]
        else:
            cc_edges=[edge_key(cycle_or_chain[i], cycle_or_chain[(i+1)]) for i in range(len(cycle_or_chain)-1)]
        pure_faces=[edge_to_face_dic[edge] for edge in cc_edges]
        if len(pure_faces)!=len(edge_to_face_dic.keys()):
            nonmani_vert=True
        neighbor_points=list(graph.nodes())

        self.verts[vert_index].NM_vert=nonmani_vert
        self.verts[vert_index].neighbor_points=neighbor_points
        self.verts[vert_index].pure_faces=pure_faces
        return nonmani_vert, pure_faces, type

    def get_edge_to_face_dic(self, face_list, triangles, vert_index):
        ret_dic={}
        for face_index in face_list:
            edge=[x for x in triangles[face_index] if x !=vert_index]
            ret_dic[edge_key(edge[0], edge[1])]=face_index
        return ret_dic

    def first_process_NM(self, triangles):
        NM_verts=[]
        mani_verts=[]
        NM_verts_cycle=[]
        for face_index, face in enumerate(triangles):
            self.verts[face[0]].neighbor_faces.append(face_index)
            self.verts[face[1]].neighbor_faces.append(face_index)
            self.verts[face[2]].neighbor_faces.append(face_index)
        # 1. init edge graph, and filter manifold/non-manifold vertex
        max_edge_num=0
        for vert_index in range(len(self.verts)):
            face_list=self.verts[vert_index].neighbor_faces
            if not face_list:
                continue
            if len(face_list)>max_edge_num:
                max_edge_num=len(face_list)
            if len(face_list) > self.NM_edge_graph_limit:
                raise Exception(f'[NM] edge graph edges exceed limit: {len(face_list)} > {self.NM_edge_graph_limit}')
            edge_to_face_dic=self.get_edge_to_face_dic(face_list, triangles, vert_index)
            nonmani_vert, pure_faces, cycle_or_chain=self.NM_init_edge_graph(vert_index, edge_to_face_dic)
            if not nonmani_vert:
                mani_verts.append(vert_index)
            else:
                NM_verts.append(vert_index)
                if cycle_or_chain == 'cycle':
                    NM_verts_cycle.append(vert_index)
        
        self.non_manifold_vertex_cnt+=len(NM_verts)
        if len(NM_verts)!=0:
            self.non_manifold=True
        if len(NM_verts)>self.NM_nonmani_vert_limit:
            raise Exception(f'[NM] NM_verts exceed limit: {len(NM_verts)} > {self.NM_nonmani_vert_limit}')
        if self.verbose:
            print(f'mani/nonmani/nonmani_cycle vertex: {len(mani_verts)}/{len(NM_verts)}/{len(NM_verts_cycle)}')
            print(f'max edge num: {max_edge_num}')
        self.max_edge_num=max_edge_num
        temp_CC_id=-1
        # 2. grouping cc for mainfold vertex, init unionfind
        for mani_vert_index in mani_verts:
            if self.verts[mani_vert_index].pre_CC_id is not None:
                continue
            temp_CC_id+=1
            self.verts[mani_vert_index].pre_CC_id=temp_CC_id
            cc_queue=[ele for ele in self.verts[mani_vert_index].neighbor_points]
            cc_queue=[ele for ele in cc_queue if ele not in NM_verts]
            cc_queue=[ele for ele in cc_queue if self.verts[ele].pre_CC_id is None]
            while cc_queue:
                first=cc_queue.pop(0)
                if self.verts[first].pre_CC_id is not None:
                    if self.verts[first].pre_CC_id != temp_CC_id:
                        raise Exception('[NM] pre_CC_id wrong')
                    continue
                if self.verts[first].NM_vert:
                    continue
                self.verts[first].pre_CC_id=temp_CC_id
                next_queue_all=[ele for ele in self.verts[first].neighbor_points if ele not in NM_verts]
                next_queue=[ele for ele in next_queue_all if self.verts[ele].pre_CC_id is None]
                next_queue_debug=[self.verts[ele].pre_CC_id for ele in next_queue_all if self.verts[ele].pre_CC_id is not None]
                if len(set(next_queue_debug)) >=2:
                    raise Exception('[NM] pre_CC_id wrong')
                cc_queue+=next_queue
        if self.verbose:
            print(f'max cc_id_pre: {temp_CC_id}')
        self.uf=UnionFind(temp_CC_id+1) # but some cc_id are equal, manage them via unionfind
        # 3. face classify based on vert CC
        triangle_CC_list=[-1 for ele in triangles]
        for mani_vert_index in mani_verts:
            pure_faces=self.verts[mani_vert_index].pure_faces
            for face_index in pure_faces:
                temp_face_cc=triangle_CC_list[face_index]
                if temp_face_cc!=-1 and temp_face_cc!=self.verts[mani_vert_index].pre_CC_id:
                    raise Exception('[NM] face cc wrong')
                triangle_CC_list[face_index]=self.verts[mani_vert_index].pre_CC_id
        if self.verbose:
            print(f'there are {triangle_CC_list.count(-1)} faces do not have cc')
        self.triangle_CC_list=triangle_CC_list
        # process non manifold vertex
        # (4). gather nonmanifold vertex
        # NM_group_list=[]
        # grouped_NM_verts=[]
        # for NM_vert_ind in NM_verts:
        #     if NM_vert_ind in grouped_NM_verts:
        #         continue
        #     grouped_NM_verts.append(NM_vert_ind)
        #     NM_group_list.append([NM_vert_ind])
        #     nm_queue=[point for point in self.verts[NM_vert_ind].neighbor_points if point in NM_verts and point not in grouped_NM_verts]
        #     while nm_queue:
        #         first=nm_queue.pop()
        #         if first in grouped_NM_verts:
        #             continue
        #         if first not in NM_verts:
        #             continue
        #         grouped_NM_verts.append(first)
        #         NM_group_list[-1].append(first)
        #         append_queue=[point for point in self.verts[first].neighbor_points if point in NM_verts and point not in grouped_NM_verts]
        #         nm_queue+=append_queue
        # if len(grouped_NM_verts)!=len(NM_verts) or len(set(grouped_NM_verts))!=len(NM_verts):
        #     raise Exception('[NM] NM group wrong')
        # print(f'NM group list ({len(NM_group_list)}): {NM_group_list}')
        # [5]. select first NM vertex to process
        next_nonmani=None
        if NM_verts_cycle:
            next_nonmani=NM_verts_cycle[0]
        elif NM_verts:
            next_nonmani= NM_verts[0]

        # 6. process selected NM vertex
        influenced_neighbor_queue=[]
        while True:
            if not NM_verts:
                break
            # print(f'processing vertex {next_nonmani}')
            NM_face_group=self.NM_process_one(next_nonmani, triangles)

            root_list=self.union_f_cc(NM_face_group) # merge cc
            # 7. new generate vertex for face
            triangles, influenced_neighbor=self.generate_new_vertex_for_NM(NM_face_group, next_nonmani, triangles)
            # if not influenced_neighbor and self.debugging:
            #     print(f'vertex: {next_nonmani} auto fixed')
            influenced_neighbor_queue+=influenced_neighbor
            # 8. update face cc
            self.update_face_CC(NM_face_group, root_list) # assign cc for face
            # 9. get face neighbor vertex list to decide the next one to process
            NM_verts, next_nonmani, influenced_neighbor_queue=self.find_next_nonmanivert(next_nonmani, NM_verts, influenced_neighbor_queue)

        return triangles
        

    def find_next_nonmanivert(self, last_index, NM_verts, influenced_neighbor_queue):
        NM_verts=[x for x in NM_verts if x!=last_index] # pop last index since it is already manifold
        if not NM_verts:
            return NM_verts, None, []
            
        while influenced_neighbor_queue:
            next_one=influenced_neighbor_queue.pop()
            if self.verts[next_one].NM_process:
                continue
            if next_one not in NM_verts:
                continue
            return NM_verts, next_one, influenced_neighbor_queue

        return NM_verts, NM_verts[0], influenced_neighbor_queue

    def update_face_CC(self, NM_face_group, root_list):
        if len(root_list)!=len(NM_face_group):
            raise Exception('no equal')
        for index in range(len(root_list)):
            face_group=NM_face_group[index]
            group_root=root_list[index]
            if group_root == -1:
                new_cc_id = self.uf.extend()
                for face in face_group:
                    self.triangle_CC_list[face]=new_cc_id
            else:
                for face in face_group:
                    if self.triangle_CC_list[face]==-1:
                        self.triangle_CC_list[face]=group_root



    
    def max_cycle_rule(self, component, edge_to_face_dic):  
        # 当图中只有一条路径时，返回最长路径  
        if not nx.is_tree(component):  
            # 查找所有的简单环  
            cycles = list(nx.simple_cycles(component.to_directed()))  
            cycles = [cycle for cycle in cycles if len(cycle)!=2]
            if cycles:  
                max_cycle = max(cycles, key=len)  
                if len(cycles) ==1:
                    return max_cycle, 'cycle'
                else:
                    # 如果有多个环，选择已有CC数量最少的环，尽量不合并CC
                    cycle_CC_num=[]
                    cycle_len=[len(cycle) for cycle in cycles]
                    final_cycle_index=None
                    min_CC_num=1000
                    max_len=0
                    for cycle in cycles:
                        cycle_edge=[edge_key(cycle[i%len(cycle)], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))]
                        cycle_face=[edge_to_face_dic[edge] for edge in cycle_edge]
                        cycle_cc=[self.triangle_CC_list[face] for face in cycle_face if self.triangle_CC_list[face]!=-1]
                        cycle_CC_num.append(self.uf.group_num(cycle_cc))
                    for cycle_index, cnt in enumerate(cycle_CC_num):
                        if cnt<min_CC_num:
                            min_CC_num=cnt
                            final_cycle_index=cycle_index
                            max_len=cycle_len[cycle_index]
                        if cnt==min_CC_num:
                            if cycle_len[cycle_index] > max_len:
                                final_cycle_index=cycle_index
                                max_len=cycle_len[cycle_index]
                    return cycles[final_cycle_index], 'cycle'

            raise Exception('can not find cycle ?')
    
        # 如果没有环，返回最长路径  
        longest_path = []  
        for node in component.nodes():  
            # 进行深度优先搜索找最长路径  
            paths = nx.single_source_dijkstra_path(component, node)  
            for path in paths.values():  
                if len(path) > len(longest_path):  
                    longest_path = path  
        return longest_path, 'path'  
    
    def NM_process_one(self, vert_index, triangles):
        self.verts[vert_index].NM_process=True
        edge_to_face_dic=self.get_edge_to_face_dic(self.verts[vert_index].neighbor_faces, triangles, vert_index)
        graph=nx.Graph()
        graph.add_edges_from(edge_to_face_dic.keys())
        origraph=graph.copy()
        cycle_list=[]
        chain_list=[]
        single_components=[]
        while True:  
            # 找到所有连通分量  
            components =  list(nx.connected_components(graph))

            if len(components) == 0:  
                break  

            for component_nodes in components:  
                # 提取出连通分量  
                if len(component_nodes)==1:
                    single_components.append(list(component_nodes))
                    graph.remove_nodes_from(list(component_nodes))
                    continue
                component = graph.subgraph(component_nodes)  

                cycle_or_path, type_ = self.max_cycle_rule(component, edge_to_face_dic)  
            
                if cycle_or_path:  
                    # print(f"找到的{type_}: {cycle_or_path}")  
                    if type_ == "cycle":
                        cycle_list.append(cycle_or_path)
                    else:
                        chain_list.append(cycle_or_path)
                    graph.remove_nodes_from(cycle_or_path)
        # check cycle
        f_CC_cycle=[]
        outcycle_edges=[]
        for cycle in cycle_list:
            cycle_edge_list=[edge_key(edge[0], edge[1]) for edge in origraph.edges(cycle)]
            if len(cycle_edge_list)!=len(cycle):
                # one or more edge connect cycle
                cycle_edge=[edge_key(cycle[i%len(cycle)], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))]
                f_CC_cycle.append([edge_to_face_dic[edge] for edge in cycle_edge])
                for edge in cycle_edge_list:
                    if edge not in cycle_edge and edge[0] in cycle and edge[1] in cycle:
                        # edge is in-cycle edge
                        f_CC_cycle.append([edge_to_face_dic[edge]])
                    elif edge not in cycle_edge:
                        # edge is out-cycle edge, it depends
                        outcycle_edges.append(edge)
            else:
                f_CC_cycle.append([edge_to_face_dic[edge] for edge in cycle_edge_list])
    
        # check chain, only have to judge start or end edge should be included to CC
        f_CC_chain=[]
        outchain_edges=[]
        for chain in chain_list:
            chain_edge_all=[edge_key(edge[0], edge[1]) for edge in origraph.edges(chain)]
            chain_edge = [edge_key(chain[i], chain[(i+1)]) for i in range(len(chain)-1)]
            for edge in chain_edge_all:
                if edge not in chain_edge:
                    outchain_edges.append(edge)

        for chain in chain_list:
        
            chain_edge_all=[edge_key(edge[0], edge[1]) for edge in origraph.edges(chain)]
            chain_edge = [edge_key(chain[i], chain[(i+1)]) for i in range(len(chain)-1)]
            start_edges=[edge_key(edge[0], edge[1]) for edge in origraph.edges(chain[:1]) if edge_key(edge[0], edge[1]) not in chain_edge]
            end_edges=[edge_key(edge[0], edge[1]) for edge in origraph.edges(chain[-1:]) if edge_key(edge[0], edge[1]) not in chain_edge]
            if len(start_edges)<=1:
                chain_edge+=start_edges       
            else:
                chain_edge+=start_edges[:1]
            if start_edges:
                outchain_edges=[x for x in outchain_edges if x!=start_edges[0]] # 因为链头的第一条edge已经添加到chain_edge
                outcycle_edges=[x for x in outcycle_edges if x!=start_edges[0]]

            if len(end_edges)<=1:
                chain_edge+=end_edges              
            else:
                chain_edge+=end_edges[:1]
            if end_edges:
                outchain_edges=[x for x in outchain_edges if x!=end_edges[0]]
                outcycle_edges=[x for x in outcycle_edges if x!=end_edges[0]]
        
            f_CC_chain.append([edge_to_face_dic[edge] for edge in chain_edge])

        # check single node
        f_CC_singlenode=[]
    
        out_edges_all=outcycle_edges+outchain_edges
        if out_edges_all:
            out_edges_all=set(out_edges_all)
            for edge in out_edges_all:
                f_CC_singlenode.append([edge_to_face_dic[edge]])


        f_CC_all=f_CC_cycle + f_CC_chain + f_CC_singlenode
    
        ret_face_all=[]
        for face_group in f_CC_all:
            ret_face_all+=face_group
        
        # assert edge_cnt==len(edge_to_face_dic.keys())
        if not len(set(ret_face_all)) ==len(edge_to_face_dic.keys()):
            raise Exception('non-manifold wrong')
            # ipdb.set_trace()
        if not len(set(ret_face_all)) == len(ret_face_all):
            # save_graph(origraph, 'debug_output/debug_dir/graph_debug1.png')
            # import ipdb;ipdb.set_trace()
            raise Exception('non-manifold wrong')
            # ipdb.set_trace()
        if not f_CC_all:
            raise Exception('non-manifold wrong')
            # ipdb.set_trace()
        return f_CC_all

    def union_f_cc(self, f_CC_all):
        root_list=[]
        for face_group in f_CC_all:
            temp_CC_view_nonge1=[self.triangle_CC_list[face] for face in face_group if self.triangle_CC_list[face]!=-1]
            if not temp_CC_view_nonge1:
                root_list.append(-1)
                continue

            root=self.uf.union_group(temp_CC_view_nonge1)   
            root_list.append(root)

        return root_list


    def __del__(self):  
        for v in self.verts:  
            del v  # 在 Python 中，引用计数自动管理内存  
        for f in self.faces:  
            for e in f.half_edges:  
                del e  
            del f