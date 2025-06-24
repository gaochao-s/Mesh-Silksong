import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import trimesh
import os
from collections import defaultdict
from datetime import datetime
import logging
### metadata

class MetaData:
    def __init__(self, **kwargs):
        self.version=kwargs.get('version')
        # origin information
        self.origin_path=kwargs.get('origin_path') # .obj file ?
        self.other_path=kwargs.get('other_path') # related point cloud, image
        self.vert_num=0 # origin input of edgeRunner process
        self.face_num=0 # origin input of edgeRunner process
        # save path
        self.M1_path=kwargs.get('M1_path')
        self.M2_path=kwargs.get('M2_path')
        self.M3_path=kwargs.get('M3_path')
        # processing information
        self.merge_repeat_verts_num=0
        self.replace_facevert_num=0
        self.degraded_face_num=0
        self.move_repeat_face_num=0
        self.flipped_face_cnt=0
        self.new_generated_verts=0
        self.non_manifold_new_gen=0
        self.non_manifold_vert_cnt=0
        self.non_manifold_process_time=0
        self.not_success_flip_face=0
        self.CC_invalid_verts=0
        self.CC_invalid_faces=0
        self.max_edge_num=0
        # post process of Mesh
        self.vert_num_p=0
        self.face_num_p=0
        self.face_type=kwargs.get('face_type')
        # METO information
        
        self.resolution = kwargs.get('resolution')
        self.OP_num =  6
        self.CC_num = 0 # connect component
        self.CC_num_valid=0
        self.CC_num_pre=0
        self.CC_num_pre_all=0
        self.start_hes_list=[]
        self.xyz_order='y-z-x'

        self.max_lv=0
        self.max_l=0

        
        self.special_type_info={}
        self.token_statistic={}
        self.CC_max_layer=[]

        self.token_length=0
        self.token_length_v1=0

        self.decoded_vert_num=0
        self.decoded_face_num=0
        self.compression_rate=None
        self.compression_rate_v1=None

    def get_compression_rate(self):
        self.compression_rate=f'{100 * self.token_length / (9 * self.face_num_p):.2f} %'

    def get_compression_rate_v1(self):
        self.compression_rate_v1=f'{100 * self.token_length_v1 / (9 * self.face_num_p):.2f} %'

    def info_to_dict(self):
        return self.__dict__
    
    def save_meta(self, save_path):
        data=self.info_to_dict()
        with open(save_path, 'w') as json_file:  
            json.dump(data, json_file, indent=4)


### load, normalize, save
def init_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # write to file
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # print to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger


def sample_save_pc(vertices, faces, save_path, pc_reso=4096):
    mesh_pc=trimesh.Trimesh(vertices=vertices, faces=faces)
    pc=mesh_pc.sample(pc_reso)
    point_cloud = trimesh.PointCloud(pc)   
    point_cloud.export(save_path) 

# read quad/triangles obj
def load_mesh_modify(file_path, bound=0.95):
    vertices = []  
    faces = []  

    with open(file_path, 'r') as file:  
        for line in file:  
              
            parts = line.strip().split()  
            if not parts:  
                continue  

            if parts[0] == 'v':  
                vertex = [float(coord) for coord in parts[1:4]]  
                vertices.append(vertex)  
            elif parts[0] == 'f':  
                
                face = [int(vertex.split('/')[0]) - 1 for vertex in parts[1:]]  
                faces.append(face)  

    for i in range(len(faces)):
        face=faces[i]
        if len(face) == 3:
            faces[i]=face+[-1]

    vertices_np = np.array(vertices)  
    faces_np = np.array(faces)  
    vertices_np = normalize_mesh(vertices_np, bound=bound)
    
    return vertices_np, faces_np  

def load_mesh(path, bound=0.95, clean=True):
    # use trimesh to load glb
    _data = trimesh.load(path)
    
    if isinstance(_data, trimesh.Scene):
        # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
        _concat = []
        # loop the scene graph and apply transform to each mesh
        scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
        for k, v in scene_graph.items():
            name = v['geometry']
            if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                transform = v['transform']
                _concat.append(_data.geometry[name].apply_transform(transform))
        _mesh = trimesh.util.concatenate(_concat)
    else:
        _mesh = _data
    
    vertices = _mesh.vertices
    faces = _mesh.faces
    
    # normalize
    vertices = normalize_mesh(vertices, bound=bound)

    # clean
    if clean:
        from kiui.mesh_utils import clean_mesh
        # only merge close vertices
        vertices, faces = clean_mesh(vertices, faces, v_pct=1, min_f=0, min_d=0, remesh=False)

    return vertices, faces

def load_mesh_nonorm(path):
    # path: string for local/s3 path
    
    _data = trimesh.load(path)

    # always convert scene to mesh, and apply all transforms...
    if isinstance(_data, trimesh.Scene):
        # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
        _concat = []
        # loop the scene graph and apply transform to each mesh
        scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
        for k, v in scene_graph.items():
            name = v['geometry']
            if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                transform = v['transform']
                _concat.append(_data.geometry[name].apply_transform(transform))
        _mesh = trimesh.util.concatenate(_concat)
    else:
        _mesh = _data
    
    vertices = _mesh.vertices
    faces = _mesh.faces

    return vertices, faces


def sort_mesh(vertices, faces):
    # sort vertices
    sort_inds = np.lexsort((vertices[:, 0], vertices[:, 2], vertices[:, 1])) # [N], sort in y-z-x order (last key is first sorted)
    vertices = vertices[sort_inds]

    # re-index faces
    inv_inds = np.argsort(sort_inds)
    faces = inv_inds[faces]

    # cyclically permute each face's 3 vertices, and place the lowest vertex first
    start_inds = faces.argmin(axis=1) # [M]
    all_inds = start_inds[:, None] + np.arange(3)[None, :] # [M, 3]
    faces = np.concatenate([faces, faces[:, :2]], axis=1) # [M, 5], ABC --> ABCAB
    faces = np.take_along_axis(faces, all_inds, axis=1) # [M, 3]

    # sort among faces (faces.sort(0) will break each face, so we have to sort as list)
    faces = faces.tolist()
    faces.sort()
    faces = np.array(faces)

    return vertices, faces

def write_obj(vertices, faces, filename):
    
    directory = os.path.dirname(filename)  
    if not os.path.exists(directory) and directory != '':  
        os.makedirs(directory)
    with open(filename, 'w') as f:
        for v in vertices:
            
            if v.shape[0]==3:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            elif v.shape[0]==6:
                f.write(f"v {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")
            else:
                raise Exception("v shape is not 3 or 6")
        for face in faces:
            if face[-1] == -1:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            elif face.shape[0] == 4 :
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")
            elif face.shape[0] == 3:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            else:
                raise Exception('face shape is not 3 or 4')

def write_ply(vertices, faces, filename):
    positions=vertices[:, :3]
    colors=vertices[:, 3:6]
    faces=faces[:, :3]
    mesh=trimesh.Trimesh(vertices=positions, faces=faces, vertex_colors=colors, process=False)
    mesh.export(filename.replace('obj','ply'))

def write_ply_fix(vertices, faces, filename):
    positions=vertices[:, :3]
    colors=vertices[:, 3:6]
    faces=faces[:, :3]
    mesh=trimesh.Trimesh(vertices=positions, faces=faces, vertex_colors=colors, process=False)
    trimesh.repair.fill_holes(mesh)  # fix hole
    trimesh.repair.fix_inversion(mesh)  # fix flipping
    trimesh.repair.fix_normals(mesh)  # fix normal
    mesh.export(filename.replace('.obj','_trimeshfix.ply'))


### token related
def write_tokens_ori(tokens, filename):
    directory = os.path.dirname(filename)  
    if not os.path.exists(directory) and directory != '':  
        os.makedirs(directory)
    with open(filename, 'w') as f:
        f.write("\ntokens:\n")
        layer_cnt=0
        for token in tokens:
            if token =="BOS":
                f.write(f"{token}#\n")
                f.write(f"TOKEN BEGIN")
            elif token == "C":
                layer_cnt=0
                f.write(f'\nC#\n ')
                f.write(f'\n[layer {layer_cnt}]\n')
            elif token == "U":
                layer_cnt+=1
                f.write(f'{token}#')
                f.write(f'\n[layer {layer_cnt}]\n')
            else:
                f.write(f"{token}#")

def trans_write_tokens(tokens, filename, engine, reso, mode):
    # if the token sequence is right (from encoder), the vertex should be decoded right, and the (x,y,z) is translated in this function
    tokens_translated=engine.translate_tokens(tokens=tokens, discrete_bins=reso, mode=mode)
    with open(filename.replace('.txt', '_trans.txt'), 'w') as f:  
        for item in tokens_translated:
            if item in ["C","U","E"]:
                f.write(f'\n{item}\n')
            elif '(' in item:
                f.write(f'# {item} ')
            else:
                f.write(f'{item} ')

def trans_write_tokens_direct(tokens, filename, engine):
    # if token sequence comes from GPT, the vertex may can not be decoded, and this function direct translate vertex token and not transform to (x,y,z)
    tokens_translated=engine.translate_tokens_direct(tokens=tokens)
    with open(filename.replace('.txt', '_transD.txt'), 'w') as f:  
        for item in tokens_translated:
            if item in ["C","U","E"]:
                f.write(f'\n{item}\n')
            else:
                f.write(f'{item} ')

def trans_compare_write_tokens(tokens, tokens_gt, filename, engine):
    # compare tokens from GPT and ground truth
    tokens_translated=engine.trans_compare_tokens(tokens, tokens_gt)
    with open(filename.replace('.txt', '_transD_compare.txt'), 'w') as f:  
        for item in tokens_translated:
            if any(ele in item for ele in ["C--","U--","E--","=C","=U","=E"]):
                f.write(f'\n{item}\n')
            elif 'longer' in item:
                f.write(f'\n{item}\n')
            else:
                f.write(f'{item} ')



### Graph
def save_graph(G, path):
    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=200, font_size=3, font_color='black')

    plt.title("Graph from Given Edges")
    plt.savefig(path)



def edges_to_faces(edges):
    # avoid key checking
    adj_dict = defaultdict(set)
    # accelerate
    edges_set = set()
    
    for a, b in edges:
        adj_dict[a].add(b)
        adj_dict[b].add(a)
        edges_set.add((min(a, b), max(a, b)))
    
    faces = []
    used_faces = set()
    
    for a, b in edges_set:

        common_neighbors = adj_dict[a] & adj_dict[b]
        
        for c in common_neighbors:
            
            if ((min(b, c), max(b, c)) in edges_set and 
                (min(a, c), max(a, c)) in edges_set):
                
                face = normalize_face(a, b, c)
                if face not in used_faces:
                    faces.append(list(face))
                    used_faces.add(face)
    
    return faces



### Normalize
def normalize_face(a, b, c):
    """min vertex index first"""
    min_vertex = min(a, b, c)
    if min_vertex == a:
        return (a, b, c) if b < c else (a, c, b)
    elif min_vertex == b:
        return (b, c, a) if c < a else (b, a, c)
    else:  # min_vertex == c
        return (c, a, b) if a < b else (c, b, a)

def normalize_mesh(vertices, bound=0.95):
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices

### Quick demo

def quick_demo(name):
    if name == 'plane':        
        # plane of two triangles
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    elif name == 'tetrahedron':
        # tetrhedron
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
    elif name == 'cube':
        # cube
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0], [4, 7, 6], [4, 6, 5]], dtype=np.int32)
    elif name == 'see':
        # a simple case that encodes to SEE
        vertices = np.array([[0.5, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1]], dtype=np.int32)
    elif name == 'lrlre':
        # a simple case that encodes to LRLRE
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 2, 0], [-2, 2, 0]], dtype=np.float32)
        vertices = normalize_mesh(vertices)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [4, 3, 5], [5, 4, 6]], dtype=np.int32)
    elif name == 'lRlre':
        # flip the second triangle of the previous case, this will lead to inconsistent face orientation
        # but our algorithm should be able to detect and correct it!
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 2, 0], [-2, 2, 0]], dtype=np.float32)
        vertices = normalize_mesh(vertices)
        faces = np.array([[0, 1, 2], [0, 3, 2], [0, 3, 4], [4, 3, 5], [5, 4, 6]], dtype=np.int32)
    elif name == 'mtype':
        # m-type
        vertices = np.array([[1, 0, 0], [3, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0]])
        vertices = normalize_mesh(vertices)
        faces = np.array([[0, 1, 2], [1, 3, 2], [4, 2, 3], [5, 4, 3], [6, 4, 5], [6, 7, 4], [8, 7, 6], [8, 9, 7], [7, 9, 2], [9, 0, 2]], dtype=np.int32)
    elif name == 'mtype_fake':
        # m-type
        vertices = np.array([[1, 0, 0], [3, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0]])
        vertices = normalize_mesh(vertices)
        faces = np.array([[7, 2, 4], [0, 1, 2], [1, 3, 2], [4, 2, 3], [5, 4, 3], [6, 4, 5], [6, 7, 4], [8, 7, 6], [8, 9, 7], [7, 9, 2], [9, 0, 2]], dtype=np.int32)
    elif name == 'mtype2':
        # m'-type
        vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [2, 1, 1], [2, 0, 1]])
        vertices = normalize_mesh(vertices)
        faces = np.array([[1, 0, 2], [2, 0 ,3], [2, 3, 4], [4, 3 ,5], [4, 5, 1], [1, 5, 0]], dtype=np.int32)
    elif name == 'torus':
        # m'-type
        vertices = np.array([
        [2, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0],
        [2, 0, 1], [2, 1, 1], [4, 1, 1], [3, 2, 1], [4, 3, 1], [2, 3, 1], [1, 2, 1], [0, 3, 1], [0, 1, 1],
        ])
        vertices = normalize_mesh(vertices)
        faces = np.array([
        [1, 2, 0], [2, 1, 3], [2, 3, 4], [4, 3, 5], [3, 6, 5], [5, 6, 7], [6, 8, 7], [1, 8, 6], [1, 0, 8],
        [9, 11, 10], [12, 10, 11], [13, 12, 11], [14, 12, 13], [14, 15, 12], [16, 15, 14], [16, 17, 15], [15, 17, 10], [17, 9, 10],
        [8, 0, 17], [9, 17, 0], [9, 0, 2], [11, 9, 2], [11, 2, 4], [13, 11, 4], [13, 4, 5], [14, 13, 5], [14, 5, 7], [16, 14, 7], [16, 7, 8], [17, 16, 8],
        [10, 1, 6], [15, 10, 6], [12, 3, 1], [10, 12, 1], [15, 6, 3], [12, 15, 3],], dtype=np.int32)
    elif name == 'torus_fake':
        # m'-type fake
        vertices = np.array([
        [2, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0],
        [2, 0, 1], [2, 1, 1], [4, 1, 1], [3, 2, 1], [4, 3, 1], [2, 3, 1], [1, 2, 1], [0, 3, 1], [0, 1, 1],])
        vertices = normalize_mesh(vertices)
        faces = np.array([
        [0, 2, 1], [3, 1, 2], [4, 3, 2], [5, 3, 4], [5, 6, 3], [7, 6, 5], [7, 8, 6], [6, 8, 1], [8, 0, 1],
        [9, 11, 10], [12, 10, 11], [13, 12, 11], [14, 12, 13], [14, 15, 12], [16, 15, 14], [16, 17, 15], [15, 17, 10], [17, 9, 10],
        [0, 8, 17], [0, 17, 9], [2, 0, 9], [2, 9, 11], [4, 2, 11], [4, 11, 13], [5, 4, 13], [5, 13, 14], [7, 5, 14], [7, 14, 16], [8, 7, 16], [8, 16, 17],
        # [6, 1, 10], [6, 10, 15], [1, 3, 12], [1, 12, 10], [3, 6, 15], [3, 15, 12],
        ], dtype=np.int32)
    elif name == 'sphere':
        # sphere
        mesh = trimesh.creation.icosphere(subdivisions=2)
        vertices = mesh.vertices
        vertices = normalize_mesh(vertices)
        faces = mesh.faces
    elif name == 'annulus':
        # annulus
        mesh = trimesh.creation.annulus(0.5, 1, 1)
        vertices = mesh.vertices
        vertices = normalize_mesh(vertices)
        faces = mesh.faces

    return vertices, faces

