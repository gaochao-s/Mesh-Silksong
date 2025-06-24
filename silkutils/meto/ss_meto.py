
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from meshdata.mesh_structure import Mesh, Vertex
from meshdata.mesh_io import MetaData, write_ply, edges_to_faces
from meto.mathutils import generate_combination_mappings
from math import comb
import random
import os

from meto.decode_utils import decode_layer_face, decode_inlayer_connect_faces, save_matrix_to_txt
from meto.decode_utils_fix import manifold_fix
import ipdb

######################
# [NOTIFICATION] there are some difference of the naming of layer adjacency matrices between code and paper
# code: inlayer matrix  -------> paper: self-layer matrix
# code: outlayer matrix -------> paper: between-layer matrix
######################

class Engine_SilkSong:  
    class OP:
        OP_C = 0  # begin of CC
        OP_U = 1  # up-layer
        OP_E = 2  # CC end
        OP_NUM = 3  # control token num

    def __init__(self, discrete_bins=256, verbose=False, debugging=False, meta_init_data={}, structure_limit={}):  
        self.discrete_bins = discrete_bins  
        self.verbose = verbose  
        self.debugging = debugging

        if self.debugging:
            self.verbose=True

        # meta information
        self.meta_data=MetaData(**meta_init_data)
        self.meta_data.resolution=discrete_bins
        self.meta_data.OP_num=self.OP.OP_NUM

        # for avoiding long time processing
        self.NM_max_edge_graph=structure_limit.get('NM_max_edge_graph', 50)
        self.NM_max_nonmani_verts=structure_limit.get('NM_max_nonmani_verts', 300)
        self.min_CC_face=structure_limit.get('min_CC_face', 3)
        self.face_num_p_limit=structure_limit.get('max_face_num_p', 12000)

        
        # results holder  
        self.mesh = None  
        self.CC_num=None
        self.non_manifold=None

        self.CC_tokens = []  # store tokens for each connected component  
        self.tokens=[] # not encode vertex
        self.tokens_encode=[] # encode vertex for training

        
        self.CC_max_layer=[]
        self.CC_inlayer_matrix=[]
        self.CC_outlayer_matrix=[]
        self.CC_layer_queue=[]

        self.control_list=["C","U","E"]
        self.layer_vertex_limit=structure_limit.get('layer_vertex_limit', 200)
        self.layer_number_limit=structure_limit.get('layer_number_limit', 200)
        self.in_layer_window=8 # 2^8=256
        self.out_layer_window=5 # stars and bars question
        # word table for self-layer-matrix in paper
        self.inlayer_emb_num=self.layer_vertex_limit+2**(self.in_layer_window)
        
        self.init_outlayer_combination()
        self.num_word_table=None
        

    def set_outlayer_window(self, winsize):
        self.out_layer_window=winsize
        self.init_outlayer_combination()

    def init_outlayer_combination(self):
        W=self.out_layer_window
        self.situation_nums=[1,  1, W-1, comb(W-1,2), comb(W-1,3),     W-1, comb(W-1, 2)]
        self.situation_bases=[0, 1, 2,   W+1,         W+1+comb(W-1,2), W+1+comb(W-1,2)+comb(W-1,3), W+1+comb(W-1,2)+comb(W-1,3)+W-1]
        self.all_situation_num=0
        for ele in self.situation_nums:
            self.all_situation_num+=ele
        
        self.situation_k=[0,0,1,2,3,1,2]
        comb_to_index_1, index_to_comb_1=generate_combination_mappings(self.out_layer_window-1, 1)
        comb_to_index_2, index_to_comb_2=generate_combination_mappings(self.out_layer_window-1, 2)
        comb_to_index_3, index_to_comb_3=generate_combination_mappings(self.out_layer_window-1, 3)
        self.comb_to_indexs=[comb_to_index_1, comb_to_index_2, comb_to_index_3] # dic
        self.index_to_combs=[index_to_comb_1, index_to_comb_2, index_to_comb_3]
        # word table for between-layer-matrix in paper
        self.outlayer_emb_num=self.layer_vertex_limit*self.all_situation_num
        # word table for totally word table of 2 layer adjacent matrix
        self.topology_num=self.inlayer_emb_num+self.outlayer_emb_num
        # word table for all
        # self.num_word_table = self.blocks**3 + self.offsets**3 + 3 + self.topology_num 
    
    def comb_offset(self, situation, b_list):
        if situation in [1, 2]:
            return 0
        
        b_comb=tuple(b_list)
        if situation in [3, 6]:
            return self.comb_to_indexs[0][b_comb]
        if situation in [4, 7]:
            return self.comb_to_indexs[1][b_comb]
        else:
            return self.comb_to_indexs[2][b_comb]

    def get_comb_index(self, situation, b_list):
        offset=self.comb_offset(situation, b_list)
        base=self.situation_bases[situation-1]
        return offset+base
    
    
    def get_index_comb(self, comb_index):
        if comb_index==0:
            return [1, []]
        if comb_index==1:
            return [2, []]
        current_offset=comb_index
        situation=None
        for index, num in enumerate(self.situation_nums):
            if current_offset-num<0:
                break
            current_offset=current_offset-num
            situation=index+2
        k=self.situation_k[situation-1]
        if situation in [3, 6]:
            b_list=self.index_to_combs[0][current_offset]
            return [situation, b_list]
        elif situation in [4, 7]:
            b_list=self.index_to_combs[1][current_offset]
            return [situation, b_list]
        else:
            b_list=self.index_to_combs[2][current_offset]
            return [situation, b_list]

    def b_list_to_window(self, situation, b_list):
        W=self.out_layer_window
        window=[]
        if situation == 1:
            window=[1]*W
        elif situation ==2:
            window=[0]*W
        elif situation ==3:
            b1=b_list[0]
            window=[1]*b1 + [0]*(W-b1)
        elif situation ==4:
            b1,b2=b_list
            window=[1]*b1+[0]*(b2-b1)+[1]*(W-b2)
        elif situation==5:
            b1,b2,b3=b_list
            window=[1]*b1+[0]*(b2-b1)+[1]*(b3-b2)+[0]*(W-b3)
        elif situation==6:
            b1=b_list[0]
            window=[0]*b1+[1]*(W-b1)
        else:
            b1,b2=b_list
            window=[0]*b1+[1]*(b2-b1)+[0]*(W-b2)
        return window

    def new_one_init(self):
        # results holder  
        self.mesh = None  
        self.CC_num=None
        self.non_manifold=None

        # self.meta_data=MetaData(**{})
        # self.meta_data.resolution=self.discrete_bins
        # self.meta_data.OP_num=self.OP.OP_NUM

        self.CC_tokens = []   
        self.tokens=[]
        self.tokens_encode=[] 
        
        self.CC_max_layer=[]

        self.CC_inlayer_matrix=[]
        self.CC_outlayer_matrix=[]
        self.CC_layer_queue=[]
        

    def get_metaData(self):
        return self.meta_data

    # map relative coordinate to positive value  
    def offset_coord(self, x):  
        return x + self.OP.OP_NUM  + self.topology_num

    def restore_coord(self, x):  
        return x - self.OP.OP_NUM - self.topology_num
    
    def set_layer_info(self, layer_id_list, layer_num):
        for id in layer_id_list:
            self.mesh.verts[id].bfs_layer=layer_num
            self.mesh.verts[id].bfs_mask=True

    def get_start_he_CC(self):
        CC_first_face=[]
        CC_start_he_list=[]
        for CC_face_list in self.mesh.CC_faces:
            CC_first_face.append(CC_face_list[0])
        for face in CC_first_face:
            FN=face.vert_FN
            he_list=[]
            for i in range(FN):
                he=face.half_edges[i]
                he_list.append(he)
            he_list.sort(key=lambda f:f)
            start_he=he_list[0]
            CC_start_he_list.append([start_he.s.i, start_he.e.i])
            # print(f"[CC y-z-x start_he]: {start_he.s.i}---->{start_he.e.i}")

        # print(CC_start_he_list)
        

        return CC_start_he_list
    
    def get_start_he_CC_specify(self, s, e): # for debugging
        # print(f"[CC Specify start_he]: {s}---->{e}")
        return [[s, e]]
    
    def get_start_he_CC_random(self): # for debugging
        CC_start_he_list=[]
        for CC_he_list in self.mesh.CC_all_half_edges:
            start_he=random.choice(CC_he_list)
            CC_start_he_list.append([start_he.s.i, start_he.e.i])
            # print(f"[CC Random start_he]: {start_he.s.i}---->{start_he.e.i}")
        return CC_start_he_list
                
    def get_next_layer_vertex(self, neighbor_list):
        layer_mask=[0 if self.mesh.verts[ele].bfs_mask else 1 for ele in neighbor_list]
        if 1 not in layer_mask:
            return []
        if 0 not in layer_mask:
            return neighbor_list
        index_0=None
        avaiable_1_cnt=sum(layer_mask)
        LN=len(layer_mask)
        for index in range(LN):
            if layer_mask[index]==0:
                index_0=index
                break
        next_index=index_0
        next_layer=[]
        while True:
            next_index+=1
            if next_index%LN ==index_0:
                break
            if self.mesh.verts[neighbor_list[next_index%LN]].bfs_mask:
                if not next_layer:
                    continue
                else:
                    break
            else:
                next_layer.append(neighbor_list[next_index%LN])
        if self.debugging:
            if len(next_layer)!=avaiable_1_cnt:
                print(f'[WARNING]: used 1/avai 1 = {len(next_layer)}/{avaiable_1_cnt}')
        return next_layer
    
    def get_inlayer_matrix(self, this_layer):
        N=len(this_layer)
        neighbor_dic={}
        inlayer_matrix=np.eye(N, dtype=int)
        for index, v in enumerate(this_layer):
            neighbors=self.mesh.verts[v].neighbors
            for ele in neighbors:
                if ele in this_layer:
                    inlayer_matrix[index][this_layer.index(ele)]=1

        if not np.array_equal(inlayer_matrix, inlayer_matrix.T):
            raise Exception('in-layer matrix not symmetrix')
        return inlayer_matrix
            
    def get_outlayer_matrix(self, last_layer, this_layer):
        N=len(last_layer)
        M=len(this_layer)
        outlayer_matrix=np.zeros((M, N), dtype=int)
        for index, v in enumerate(this_layer):
            neighbors=self.mesh.verts[v].neighbors
            for ele in neighbors:
                if ele in last_layer:
                    outlayer_matrix[index][last_layer.index(ele)]=1
        return outlayer_matrix


    def encode(self, vertices, triangles, non_mani_process):  
        # build mesh encode
        
        self.new_one_init()
        
        self.mesh = Mesh(vertices=vertices, triangles=triangles, discrete_bins=self.discrete_bins, verbose=self.verbose, debugging=self.debugging, non_mani_process=non_mani_process, NM_max_edge_graph=self.NM_max_edge_graph, NM_max_nonmani_verts=self.NM_max_nonmani_verts, min_CC_face=self.min_CC_face, M2_path=self.meta_data.M2_path)  
        # face.ic = 1,2,3 ...
        # vert.CC_id = 0,1,2 ...
        parent, _=self.mesh.uf.parent_list()
        self.meta_data.max_edge_num=self.mesh.max_edge_num
        self.CC_num=self.mesh.num_components
        self.non_manifold=self.mesh.non_manifold
        self.meta_data.CC_num_pre_all=len(parent)
        self.meta_data.CC_num_pre=len(self.mesh.unique_roots)
        self.meta_data.vert_num=self.mesh.num_vertices_all
        self.meta_data.face_num=self.mesh.num_faces_all
        self.meta_data.vert_num_p=self.mesh.num_vertices_p
        self.meta_data.face_num_p=self.mesh.num_faces_p
        self.meta_data.CC_invalid_verts=self.mesh.CC_invalid_verts_cnt
        self.meta_data.CC_invalid_faces=self.mesh.CC_invalid_face_cnt

        self.meta_data.CC_num=self.CC_num
        self.meta_data.CC_num_valid=self.mesh.num_components_valid
        self.meta_data.flipped_face_cnt=self.mesh.fliped_face_cnt
        self.meta_data.not_success_flip_face=len(self.mesh.not_success_flip_face)
        self.meta_data.new_generated_verts=self.mesh.new_generated_verts
        self.meta_data.merge_repeat_verts_num=self.mesh.merge_repeat_verts_num
        self.meta_data.replace_facevert_num=self.mesh.replace_facevert_num
        self.meta_data.degraded_face_num=self.mesh.degraded_face_num
        self.meta_data.move_repeat_face_num=self.mesh.move_repeat_face_num
        self.meta_data.non_manifold_new_gen=self.mesh.non_manifold_born_vert_num
        self.meta_data.non_manifold_vert_cnt=self.mesh.non_manifold_vertex_cnt
        self.meta_data.non_manifold_process_time=-1
        self.CC_valid_mask=self.mesh.CC_mask

        # print(f'flipped {self.meta_data.flipped_face_cnt}/{self.meta_data.face_num} faces')
        # print(f'new generate {self.meta_data.new_generated_verts} verts')
        if self.debugging:
            # temp_verts=np.asarray([vert.undiscrete_vertex(color='cc_face') for vert in self.mesh.verts])
            if self.meta_data.CC_num_valid > 100:
                pass
                # if self.debugging:
                #     write_obj(temp_verts, self.mesh.triangles_process, 'debug_output/debug_dir/CC-result.obj')
                # import ipdb;
                # ipdb.set_trace()
                # raise Exception(f'[E] too many valid CC {self.meta_data.CC_num_valid}/{self.meta_data.CC_num} > 100!')
        
        if self.meta_data.face_num_p > self.face_num_p_limit:
            # pass
            raise Exception(f'[E] too many faces_p {self.meta_data.face_num_p} > 12k!')

        if self.non_manifold:
            # print(self.mesh.non_manifold_edge_stats)
            if self.verbose:
                # write_obj(temp_verts, self.mesh.triangles_process, 'debug_output/debug_dir/non-manifold-result.obj')
                print(f'non-manifold new gen: {self.meta_data.non_manifold_new_gen}')
                print(f'non-manifold vertex cnt: {self.meta_data.non_manifold_vert_cnt}')
        
        lone_vert_num=0
        for v in self.mesh.verts:
            if v.CC_id is None and v.repeat_mask == False:
                lone_vert_num+=1
                # print(f'lone vert : {v.i}')
        

        for k in range(self.CC_num):
            
            self.CC_inlayer_matrix.append([])
            self.CC_outlayer_matrix.append([])
            self.CC_layer_queue.append([])
            self.CC_tokens.append([])
            
        
        # 1. preprocess to record vertex's half edges:
        # print(f'[step 1]--------------recording verts based on half edge--------------')
        for CC_i, CCs in enumerate(self.mesh.CC_all_half_edges):
            if self.CC_valid_mask[CC_i] == False:
                continue
            for half_edge in CCs:
                start_id =half_edge.s.i
                end_id = half_edge.e.i
                if not( start_id >=0 and end_id >=0):
                    raise Exception('[E] start_id/end_id wrong')
                self.mesh.verts[start_id].start_half_edges.append(half_edge)
                self.mesh.verts[end_id].end_half_edges.append(half_edge)

        # print(f'[step 1.1]--------------setting vert bound info--------------')
        # and handle non-mainfold (repeated half edge for one vertex)
        for vert in self.mesh.verts:
            if vert.repeat_mask==True:
                continue
            if vert.CC_id is None:
                continue
            if self.CC_valid_mask[vert.CC_id] == False:
                continue
            # set the local order for each vertex;
            # as well as the bound vertex
            vert.set_bound_info()
            

        # 2. pick a start half-edge
        # print(f'[step 2]--------------picking start half edge--------------')
        # she
        CC_start_he_list = self.get_start_he_CC()
        
        if self.verbose:
            print(f'lone vert num: {lone_vert_num}')
            print(f'start_he {CC_start_he_list}')
            
        self.meta_data.start_hes_list=CC_start_he_list
        # print(f'[step 3]--------------[CC] BFS layer calculation for verts--------------')
        max_lv=0
        for CC_index in range(self.CC_num):
            if self.CC_valid_mask[CC_index] == False:
                self.CC_max_layer.append(-1)
                continue
            start_he_center_v=CC_start_he_list[CC_index][0]
            start_he_pointer_v=CC_start_he_list[CC_index][1]
            # 3. layering all verts
            self.mesh.verts[start_he_center_v].bfs_layer=0
            self.mesh.verts[start_he_center_v].bfs_order=0
            self.mesh.verts[start_he_center_v].bfs_mask=True
            
            self.CC_layer_queue[CC_index].append([start_he_center_v])
            self.CC_inlayer_matrix[CC_index].append(None)
            self.CC_outlayer_matrix[CC_index].append(None)
            self.CC_tokens[CC_index].append(["C", start_he_center_v+self.topology_num, "U"])
            layer_before=0
            layer_current=1
            
            
            while True:
                if layer_current>self.layer_number_limit:
                    raise Exception(f'[E] reach layer limit {self.layer_number_limit}')
                
                last_layer_queue=self.CC_layer_queue[CC_index][layer_before]
                next_layer_queue=[]
                for vlast in last_layer_queue:
                    next_layer_verts=self.get_next_layer_vertex(self.mesh.verts[vlast].neighbors)
                    next_layer_verts_unique=[ele for ele in next_layer_verts if ele not in next_layer_queue]
                    next_layer_queue+=next_layer_verts_unique
                if not next_layer_queue:
                    self.CC_max_layer.append(layer_before)
                    break
                if len(next_layer_queue)>max_lv:
                    max_lv=len(next_layer_queue)
                self.CC_layer_queue[CC_index].append(next_layer_queue)
                self.set_layer_info(next_layer_queue, layer_current)


                layer_current=layer_current+1
                layer_before=layer_before+1

        # check if we lost some verts
        layer_verts_cnt=0
        for CC_i, CC_lay in enumerate(self.CC_layer_queue):
            
            if self.CC_valid_mask[CC_i] == False:
                continue
            for lay in CC_lay:
                layer_verts_cnt=layer_verts_cnt+len(lay)
        
        if layer_verts_cnt != self.mesh.num_vertices_p-lone_vert_num:
            if self.debugging:
                print(f'[E] we may lost some vertex in some CC {layer_verts_cnt}!={self.mesh.num_vertices_p-lone_vert_num}')
            # ipdb.set_trace()
            # raise Exception(f'[E] we may lost some vertex in some CC {layer_verts_cnt}!={self.mesh.num_vertices_p-lone_vert_num}')
            # print('we may lost some vertex in some CC')

        self.meta_data.CC_max_layer=self.CC_max_layer
        self.meta_data.max_lv=max_lv
        self.meta_data.max_l=max(self.CC_max_layer)
        if self.debugging:
            self.M3_saving(faces=self.mesh.triangles_process)
            print(f'max lv : {max_lv}')
            print(f'max l {self.meta_data.max_l}')
        if max_lv>self.layer_vertex_limit:
            raise Exception(f'[Encoder] max lv {max_lv} > {self.layer_vertex_limit}')
        # print(f'[step 4]--------------[CC] Matrix calculation--------------')
        statistic_dic={}
        for CC_index in range(self.CC_num):
            if self.CC_valid_mask[CC_index] == False:
                continue
            max_layer=self.CC_max_layer[CC_index]
            for layer_num in range(1, max_layer+1):
                last_layer_queue=self.CC_layer_queue[CC_index][layer_num-1]
                this_layer_queue=self.CC_layer_queue[CC_index][layer_num] # M
                inlayer_matrix=self.get_inlayer_matrix(this_layer_queue) # M x M
                outlayer_matrix=self.get_outlayer_matrix(last_layer_queue, this_layer_queue) # M x N 
                inlayer_embeding=self.compress_inlayer_matrix(inlayer_matrix)
                outlayer_embeding, _=self.compress_outlayer_matrix(outlayer_matrix)
                this_layer_token=[]
                for v, i, o in zip(this_layer_queue, inlayer_embeding, outlayer_embeding):
                    this_layer_token+=[v+self.topology_num]
                    this_layer_token+=i
                    this_layer_token+=o
                    if len(i) !=1:
                        key_in=f'in_{len(i)}'
                        if key_in not in statistic_dic:
                            statistic_dic[key_in]=1
                        else:
                            statistic_dic[key_in]+=1
                    if len(o) !=1:
                        key_out=f'out_{len(o)}'
                        if key_out not in statistic_dic:
                            statistic_dic[key_out]=1
                        else:
                            statistic_dic[key_out]+=1
                if layer_num == max_layer:
                    this_layer_token+=["E"]
                else:
                    this_layer_token+=["U"]
                self.CC_tokens[CC_index].append(this_layer_token)
                self.CC_inlayer_matrix[CC_index].append(inlayer_matrix)
                self.CC_outlayer_matrix[CC_index].append(outlayer_matrix)

        # print(f'[step 5]--------------[CC] GET original token--------------')
        self.meta_data.token_statistic=statistic_dic
        if self.debugging:
            print(statistic_dic)
        for CC_index in range(self.CC_num):
            if self.CC_valid_mask[CC_index] == False:
                continue
            max_layer=self.CC_max_layer[CC_index]
            for layer_num in range(0, max_layer+1):
                self.tokens+=self.CC_tokens[CC_index][layer_num]

        return self.tokens

    
    def compress_inlayer_matrix(self, inlayer_matrix):
        inlayer_extend=np.hstack((inlayer_matrix, inlayer_matrix)) # Nx2N
        M=inlayer_matrix.shape[0]-1 # N-1
        W=self.in_layer_window
        if M < W:
            inlayer_homo_m=np.array([row[i+1:i+1+M] for i, row in enumerate(inlayer_extend)]) # N,M->N,w
            padding_columns=W-M
            inlayer_homo = np.pad(inlayer_homo_m, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)
            compress_id = [[int(''.join(map(str, row[::-1])), 2)] for row in inlayer_homo]
            compress_id = [[ele[0]+self.layer_vertex_limit] for ele in compress_id]
            return compress_id
        inlayer_homo=np.array([row[i+1:i+1+M] for i, row in enumerate(inlayer_extend)])
        compress_id = [[int(''.join(map(str, row[:W][::-1])), 2)] for row in inlayer_homo]
        compress_id = [[ele[0]+self.layer_vertex_limit] for ele in compress_id]
        if 2*W>=M:
            return compress_id
        else:
            # print(f'inlayer homo \n {inlayer_homo}')
            inlayer_homo[:, -W:]=0
            inlayer_homo[:, :W]=0
            indices_list = [list(np.where(row == 1)[0]) for row in inlayer_homo]
            for line, ele in enumerate(indices_list):
                compress_id[line]+=ele
            return compress_id
        
    def decompress_inlayer_matrix(self, compress_id):
        N=len(compress_id)
        M=N-1
        # homo_number=[ele[0]-self.layer_vertex_limit for ele in compress_id]
        homo_number=[]
        for line, ele in enumerate(compress_id):
            ele_0=ele[0]-self.layer_vertex_limit
            if ele_0<0:
                print('[WARNING] inlayer matrix first ele wrong')
                homo_number.append(ele[0])
            else:
                homo_number.append(ele_0)
        W=self.in_layer_window
        inlayer_homo_init=np.array([list(format(num, f'0{W}b')[::-1]) for num in homo_number], dtype=int)
        if M<W:
            inlayer_homo=inlayer_homo_init[:, :M]
        else:
            padding_columns=M-W
            inlayer_homo = np.pad(inlayer_homo_init, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)
            for index, ele in enumerate(compress_id):
                for ind in ele[1:]:
                    inlayer_homo[index][ind]=1
        inlayer_extend=np.eye(N, dtype=int)
        inlayer_extend=np.hstack((inlayer_extend, inlayer_extend))
        for i, row in enumerate(inlayer_homo):
            inlayer_extend[i, i+1:i+1+M]=row
        inlayer_A=inlayer_extend[:, :N]
        inlayer_B=inlayer_extend[:, N:]
        
        low_tri_B=np.tril(inlayer_B)
        inlayer_A+=low_tri_B
        inlayer_A-=np.eye(N, dtype=int)
        # fix to symm matrix
        inlayer_matrix=np.maximum(inlayer_A, inlayer_A.T)
        return inlayer_matrix


    def judge_situation(self, window):
        if 0 not in window:
            return [1, [], None]
        if 1 not in window:
            return [2, [], None]
        b1=None
        b2=None
        b3=None
        novalid_ind=None
        situation=None
        window_unique=[]
        for i in range(len(window)):
            if window[i]==1:
                if not window_unique:
                    window_unique.append(1)
                elif window_unique[-1]==1:
                    continue
                else:
                    # ...0   <-- 1
                    if not b2:
                        b2=i
                        window_unique.append(1)
                    else:
                        novalid_ind=i
                        break
            else:
                if not window_unique:
                    b1=-1
                    window_unique.append(0)
                elif window_unique[-1]==0:
                    continue
                else:
                    # ...1  <-- 0
                    window_unique.append(0)
                    if not b1:
                        b1=i
                    else:
                        b3=i
        b_list=[]
        if len(window_unique)==2:
            if window_unique[0]==0:
                situation=6
                b_list=[b2] # 1~w-1
            else:
                situation=3
                b_list=[b1] # 1~w-1
        elif len(window_unique)==3:
            if window_unique[0]==0:
                situation=7
                b_list=[b2, b3] # C w-1 ^ 2
            else:
                situation=4
                b_list=[b1, b2] # C w-1 ^ 2
        else:
            assert len(window_unique)==4
            situation=5
            b_list=[b1, b2, b3] # C w-1 ^ 3
        if None in b_list:
            raise Exception('[Encoder] None in b_list')
        return [situation, b_list, novalid_ind]
        

    def compress_window(self, next_row_pad, find_index, W):
        window=next_row_pad[find_index+1: find_index+1+W]
        situation_list=self.judge_situation(window)
        situation=situation_list[0]
        b_list=situation_list[1]
        # print(f'encode: situation{situation}, b_list{b_list}, win{window}')
        if situation_list[2] is None:
            window=[0]*W
        else:
            novalid_index=situation_list[2]
            window[:novalid_index]=[0]*novalid_index
        comb_index=self.get_comb_index(situation, b_list)
        # print(f'win fix: {window}')

        next_row_pad[find_index+1: find_index+1+W]=window
        return next_row_pad, comb_index

    def to_row_embid(self, find_index, comb_index):
        return find_index*self.all_situation_num+comb_index+self.inlayer_emb_num
    
    def from_row_embid(self, embid):
        embid=embid-self.inlayer_emb_num
        find_index = embid // self.all_situation_num
        comb_index = embid % self.all_situation_num
        return find_index, comb_index

    def compress_outlayer_matrix(self, outlayer_matrix):
        M, N = outlayer_matrix.shape # N: last layer
        W=self.out_layer_window
        outlayer_matrix_pad = np.pad(outlayer_matrix, ((0,0) ,(0, W)), mode='constant', constant_values=0)
        embids=[]
        row_all=[]
        # ipdb.set_trace()
        for line, row_pad in enumerate(outlayer_matrix_pad):
            next_row_pad=list(row_pad)
            row_result=[]
            row_embid=[]
            for find_index in range(N):
                if next_row_pad[find_index]==1:
                    next_row_pad, comb_index = self.compress_window(next_row_pad, find_index, W)
                    row_result.append([find_index, comb_index])
                    row_embid.append(self.to_row_embid(find_index, comb_index))
            embids.append(row_embid)
            row_all.append(row_result)
        # print('row result')
        # print(row_all)
        return embids, N
    
    def decompress_outlayer_matrix(self, embids, N):
        W=self.out_layer_window
        M=len(embids)
        outlayer_matrix_pad=np.zeros((M, N+W), dtype=int)
        for line, embid_row in enumerate(embids):
            for embid in embid_row:
                find_index, comb_index= self.from_row_embid(embid)
                # print(f'decode index: {find_index}; comb_index: {comb_index}')
                situation, b_list=self.get_index_comb(comb_index)
                window=self.b_list_to_window(situation, b_list)
                # print(f'decode: situation{situation}, b_list{b_list}, win{window}')
                if find_index+1>N:
                    print('[WARNING] Decoder-OutMatrix find_index+1 > N')
                    continue
                    raise Exception('[Decoder] find_index+1 > N')
                outlayer_matrix_pad[line, find_index: find_index+1+W]=[1]+window
        outlayer_matrix=outlayer_matrix_pad[:, :-W]
        return outlayer_matrix

    def get_token_classify(self):
        # 0=PAD, 1=BOS, 2=EOS, 3=C, 4=U, 5=E, [6, topo, vert]
        token_classify_dic={
            'EOS': [2],
            'C': [3],
            'U': [4],
            'E': [5],
            'In0': list(range(6+self.layer_vertex_limit, 6+self.inlayer_emb_num)),
            'In1': list(range(6, 6+self.layer_vertex_limit)),
            'Out': list(range(6+self.inlayer_emb_num, 6+self.topology_num)),
            'Block': list(range(6+self.topology_num, 6+self.topology_num+8**3)),
            'Offset': list(range(6+self.topology_num+8**3, 6+self.topology_num+8**3+16**3)),
            'vertex': list(range(6+self.topology_num, 6+self.topology_num+self.discrete_bins))
        }
        return token_classify_dic

    def get_token_map_list(self, token_classify_dic, mode):
        if mode==3:
            token_map_list=[-1]*(self.OP.OP_NUM+self.topology_num+self.discrete_bins)
        elif mode==4:
            token_map_list=[-1]*(self.OP.OP_NUM+self.topology_num+self.blocks**3+self.offsets**3)
        for key in token_classify_dic.keys():
            if key == 'EOS':
                continue
            if key in ["Block","Offset"] and mode==3:
                continue
            if key == 'vertex' and mode==4:
                continue
            value=token_classify_dic[key]
            for index in value:
                if token_map_list[index-3]==-1:
                    token_map_list[index-3]=key
                else:
                    raise Exception('why repeat')
        if -1 in token_classify_dic:
            raise Exception('why -1 still in')
        return token_map_list
    
    def get_token_map_list_GPT(self, token_classify_dic, mode):
        if mode==3:
            token_map_list=[-1]*(3+self.OP.OP_NUM+self.topology_num+self.discrete_bins)
        elif mode==4:
            token_map_list=[-1]*(3+self.OP.OP_NUM+self.topology_num+self.blocks**3+self.offsets**3)
        for key in token_classify_dic.keys():
            if key in ["Block","Offset"] and mode==3:
                continue
            if key == 'vertex' and mode==4:
                continue
            value=token_classify_dic[key]
            for index in value:
                if token_map_list[index]==-1:
                    token_map_list[index]=key
                else:
                    raise Exception('why repeat')
        token_classify_dic[0]="PAD"
        token_classify_dic[1]="BOS"
        if -1 in token_classify_dic:
            raise Exception('why -1 still in')
        return token_map_list

    def token_precheck(self, tokens, mode):
        token_classify_dic=self.get_token_classify()
        token_map_list=self.get_token_map_list(token_classify_dic, mode)

        i=1
        N=len(tokens)
        if N==0:
            raise Exception('[Decoder] token length is 0')
        if tokens[0]!=self.OP.OP_C:
            raise Exception('[Decoder] tokens[0] not C')
        while i < N-1:
            cur_token=tokens[i]
            last_token=tokens[i-1]
            cur_type=token_map_list[cur_token]
            last_type=token_map_list[last_token]
            next_token=tokens[i+1]
            next_type=token_map_list[next_token]
            if cur_type=='C':
                if next_type not in ['Block','vertex']:
                    raise Exception(f'{i}: C next wrong')
                if last_type != 'E':
                    raise Exception(f'{i}: C last not E')
            elif cur_type == 'U':
                if next_type not in ['Block','vertex']:
                    raise Exception(f'{i}: U next wrong')
                if last_type not in ['vertex','Offset','Out']:
                    raise Exception(f"{i}: U last wrong")
            elif cur_type == 'E':
                if next_type != "C":
                    raise Exception(f'{i}: E next not C')
                if last_type != "Out":
                    raise Exception(f"{i}: E last not Out")
            elif cur_type == "In0":
                if next_type not in ["In1", "Out"]:
                    raise Exception(f'{i}: In0 next wrong')
                if last_type not in ["Offset","vertex"]:
                    raise Exception(f'{i}: In0 last wrong')
            elif cur_type == "In1":
                if next_type not in ["In1", "Out"]:
                    raise Exception(f'{i}: In1 next wrong')
                if last_type not in ["In0","In1"]:
                    raise Exception(f'{i}: In1 last wrong')
            elif cur_type == "Out":
                if next_type not in ["Out", "E", "U", "vertex", "Block"]:
                    raise Exception(f'{i}: Out next wrong')
                if last_type not in ["Out", "In0", "In1","vertex","Offset"]:
                    raise Exception(f'{i}: Out last wrong')
            elif cur_type == "vertex":
                # mode == 3
                if last_type not in ["Out","C","U"]:
                    raise Exception(f'{i}: vertex last wrong')
                if not (i+1<N and i+2<N):
                    raise Exception(f'{i}: vertex not complete')
                next_1_type=token_map_list[tokens[i+1]]
                next_2_type=token_map_list[tokens[i+2]]
                if next_1_type != "vertex" or next_2_type != 'vertex':
                    raise Exception(f'{i}: vertex next 3 wrong')
                if i+3 < N:
                    next_type=token_map_list[tokens[i+3]]
                    if next_type not in ["In0", "U"]:
                        raise Exception(f'{i}: vertex next wrong')
                i=i+2
            elif cur_type == "Block":
                if last_type not in ["Out","C","U"]:
                    raise Exception(f'{i}: Block last wrong')
                if next_type not in ["Offset"]:
                    raise Exception(f'{i}: Block next not Offset')
            elif cur_type == "Offset":
                if last_type != "Block":
                    raise Exception(f'{i}: Offset last not Block')
                if next_type not in ["In0", "U"]:
                    raise Exception(f'{i}: Offset next wrong')
            else:
                raise Exception('Wrong')
                
            i+=1
                
    
    def token_encode(self, input_tokens, mode):

        self.init_BO(mode)
        last_block_id=None
        self.tokens_encode=[]
        

        for ele in input_tokens:
            if ele in self.control_list:
                if ele == "C":
                    self.tokens_encode.append(self.OP.OP_C)
                elif ele == "U":
                    self.tokens_encode.append(self.OP.OP_U)
                    last_block_id=None
                elif ele == "E":
                    self.tokens_encode.append(self.OP.OP_E)
                else:
                    raise Exception("[E] token encoding wrong")
                continue
            ele=int(ele)
            if ele < self.topology_num:
                # inlayer or outlayer
                self.tokens_encode.append(ele+self.OP.OP_NUM)
            else:
                ele -= self.topology_num
                if mode ==3:
                    self.tokens_encode+=[self.offset_coord(self.mesh.verts[ele].x), self.offset_coord(self.mesh.verts[ele].y), self.offset_coord(self.mesh.verts[ele].z)]
                elif mode==4:
                    block_id, offset_id=self.to_BO(self.mesh.verts[ele].x, self.mesh.verts[ele].y, self.mesh.verts[ele].z)
                    self.tokens_encode+=[self.offset_coord(block_id), self.offset_coord(offset_id)]
                
                else:
                    raise Exception("[E] mode not implement")
        
        
        self.meta_data.token_length=len(self.tokens_encode)
        self.meta_data.get_compression_rate()
            

        # if mode==3:
        #     self.meta_data.token_length_v1=len(self.tokens_encode)
        #     self.meta_data.get_compression_rate_v1()
            
        ret=self.tokens_encode
        

        return ret
    
    def to_BO(self, t_x, t_y, t_z):
        block_id_xyz= [t_x//self.offsets, t_y//self.offsets, t_z//self.offsets]
        block_id = block_id_xyz[0]*self.blocks**2 + block_id_xyz[1]*self.blocks + block_id_xyz[2]
        offset_id_xyz= [t_x%self.offsets, t_y%self.offsets, t_z%self.offsets]
        offset_id= offset_id_xyz[0]*self.offsets**2 + offset_id_xyz[1]*self.offsets + offset_id_xyz[2]
        offset_id += self.blocks**3
        return block_id, offset_id
    
    def from_BO(self, block_id, offset_id):
        #  block_id  block_id_xyz
        block_id_z = block_id % self.blocks
        block_id_y = (block_id // self.blocks) % self.blocks
        block_id_x = block_id // (self.blocks ** 2)
    
        #  offset_id  offset_id_xyz
        offset_id=offset_id - self.blocks**3
        offset_id_z = offset_id % self.offsets
        offset_id_y = (offset_id // self.offsets) % self.offsets
        offset_id_x = offset_id // (self.offsets ** 2)
    
        #  t_x, t_y, t_z
        t_x = block_id_x * self.offsets + offset_id_x
        t_y = block_id_y * self.offsets + offset_id_y
        t_z = block_id_z * self.offsets + offset_id_z
    
        return t_x, t_y, t_z
    
    def init_BO(self, mode):
        # 128=8*16
        # 256=?
        if mode in [2,4]:
            self.blocks=8
            self.offsets=16
            if self.blocks*self.offsets!=self.discrete_bins:
                raise Exception('[E] block * offset != resolution')
        elif mode in [1,3]:
            self.blocks=None
            self.offsets=None
        else:
            raise Exception('mode not impl')

    
    def M3_saving(self, faces):
        M3_path=self.meta_data.M3_path
        
        vertices_colored, faces = self.decode_bfs(faces)
        print(f'M3 saving to {M3_path}')
        
        write_ply(np.asarray(vertices_colored), np.asarray(faces), M3_path)

    

    def decode_bfs(self, faces):
        # mesh  
        vertices = []  
        faces = faces
        for vert in self.mesh.verts:
            
            vertices.append(vert.undiscrete_vertex(color='layer'))

        
        return vertices, faces

    
    def token1_to_vert(self, token1, layer, index, up_first):
        # for index
        
        x=self.mesh.verts[token1].x
        y=self.mesh.verts[token1].y
        z=self.mesh.verts[token1].z
        v = Vertex(x, y, z, index=index)
        v.bfs_layer=layer
        v.discrete_bins=self.discrete_bins
        return v
    
    
    def decode_next_v(self, tokens, i, layer, index, discrete_bins, mode=1):
        
        if mode==3:
            if i+2>=len(tokens):
                raise Exception("can not read next vertex mode 3")
            v_x = tokens[i]
            v_y = tokens[i+1]
            v_z = tokens[i+2]
            offset_all=self.OP.OP_NUM+self.topology_num
            if not( v_x >=offset_all and v_y >=offset_all and v_z >=offset_all):
                raise Exception('[Decoder] decode next v < offset_all')
            v = Vertex(self.restore_coord(v_x), self.restore_coord(v_y), self.restore_coord(v_z), index=index)
            v.bfs_layer=layer
            v.discrete_bins=discrete_bins
            return v, i+2
        elif mode==4:
            v_first=tokens[i]
            v_first_restore=self.restore_coord(v_first)
            if v_first_restore < self.blocks**3:
                # this is block id!
                if i+1>=len(tokens):
                    raise Exception("[Decoder] can not read next vertex mode 4")
                v_second=tokens[i+1]
                v_second_restore=self.restore_coord(v_second)
                if v_second_restore < self.blocks**3:
                    raise Exception('[Decoder] bolck_id block_id situation')
                t_x,t_y,t_z=self.from_BO(v_first_restore, v_second_restore)
                v = Vertex(t_x, t_y, t_z, index=index)
                v.bfs_layer=layer
                v.discrete_bins=discrete_bins
                self.current_block_id=v_first_restore
                return v, i+1
            else:
                # this is offset id!
                v_first_restore = self.current_block_id
                v_second=tokens[i]
                v_second_restore=self.restore_coord(v_second)

                t_x,t_y,t_z=self.from_BO(v_first_restore, v_second_restore)
                v = Vertex(t_x, t_y, t_z, index=index)
                v.bfs_layer=layer
                v.discrete_bins=discrete_bins
                
                return v, i
        else:
            raise Exception("not implement")

    def init_decoder(self):
        self.edges_decode=[]
        self.vertices_decode=[]
        self.num_vert=0

        self.num_CC_decode=0
        self.CC_decode_inlayer_matrix=[]
        self.CC_decode_outlayer_matrix=[]
        self.CC_decode_vert_queue=[]

        # layer up will clean
        self.layer_inlayer_mat_embs=[]
        self.layer_outlayer_mat_embs=[]
        self.layer_vert_current=[]
        self.layer_vert_last=[]

        self.current_layer=0

    def uplayer_clean(self):
        self.layer_inlayer_mat_embs=[]
        self.layer_outlayer_mat_embs=[]
        self.layer_vert_current=[]
        self.layer_vert_last=[]

    def decode_inlayer_matrix(self, vert_list, inlayer_matrix):
        edges = []
        for i in range(len(vert_list)):
            for j in range(i + 1, len(vert_list)):  # check up triangle of matrix
                if inlayer_matrix[i, j] == 1:
                    edges.append([vert_list[i], vert_list[j]])
        return edges
    
    def decode_outlayer_matrix(self, vert_list_last, vert_list, outlayer_matrix):
        edges = []
        for i in range(len(vert_list)):
            for j in range(len(vert_list_last)):  
                if outlayer_matrix[i, j] == 1:
                    edges.append([vert_list[i], vert_list_last[j]])
        return edges

    def decode_ori(self, tokens):  
        
        # print('-------------------------start decoding ori----------------------')
        
        self.edges_decode=[]
        self.vertices_decode=[]
        self.num_vert=0

        self.num_CC_decode=0
        self.CC_decode_inlayer_matrix=[]
        self.CC_decode_outlayer_matrix=[]
        self.CC_decode_vert_queue=[]
        i=0
        # layer up will clean
        self.layer_inlayer_mat_embs=[]
        self.layer_outlayer_mat_embs=[]
        self.layer_vert_current=[]
        self.layer_vert_last=[]

        self.current_layer=0
        
        while i<len(tokens):
            cur_token=tokens[i]
            
            if cur_token == "C":
                self.num_CC_decode+=1
                self.CC_decode_inlayer_matrix.append([])
                self.CC_decode_outlayer_matrix.append([])
                self.CC_decode_vert_queue.append([])
                self.current_layer=0
                i+=1
                continue
            if cur_token in ["U","E"]:
                
                if self.current_layer==0:
                    cur_layer_vert=self.layer_vert_current
                    self.uplayer_clean()
                    self.layer_vert_last=cur_layer_vert
                    self.current_layer+=1
                    self.CC_decode_inlayer_matrix[self.num_CC_decode-1].append(None)
                    self.CC_decode_outlayer_matrix[self.num_CC_decode-1].append(None)
                    i+=1
                    continue

                inlayer_M=len(self.layer_inlayer_mat_embs)
                outlayer_M=len(self.layer_outlayer_mat_embs)
                layer_M=len(self.layer_vert_current)
                last_layer_N=len(self.layer_vert_last)
                if layer_M!=inlayer_M or layer_M!=outlayer_M:
                    raise Exception('[Decoding] matrix dim wrong')
                
                try:
                    inlayer_matrix=self.decompress_inlayer_matrix(self.layer_inlayer_mat_embs)
                    outlayer_matrix=self.decompress_outlayer_matrix(self.layer_outlayer_mat_embs, last_layer_N)
                
                    edges_in=self.decode_inlayer_matrix(self.layer_vert_current, inlayer_matrix)
                    edges_out=self.decode_outlayer_matrix(self.layer_vert_last, self.layer_vert_current, outlayer_matrix)
                except Exception as e:
                    print(f'[Decode] {str(e)}')
                    break

                self.edges_decode+=edges_in
                self.edges_decode+=edges_out

                cur_layer_vert=self.layer_vert_current
                self.uplayer_clean()
                self.layer_vert_last=cur_layer_vert
                self.current_layer+=1

                self.CC_decode_inlayer_matrix[self.num_CC_decode-1].append(inlayer_matrix)
                self.CC_decode_outlayer_matrix[self.num_CC_decode-1].append(outlayer_matrix)
                i+=1
                continue
            
            cur_token=int(cur_token)
            if cur_token < self.topology_num: # topology token
                target_index=len(self.layer_vert_current)-1
                if cur_token<self.inlayer_emb_num: # this is inlayer emb
                    self.layer_inlayer_mat_embs[target_index].append(cur_token)
                else:
                    self.layer_outlayer_mat_embs[target_index].append(cur_token)
                i+=1
                continue
            else: # normal vertex
                cur_token=cur_token-self.topology_num
                try:
                    v_cur = self.token1_to_vert(cur_token, layer=self.current_layer, index=self.num_vert, up_first=None)
                except Exception as e:
                    print(f'[Decode] {str(e)}')
                    break
                self.num_vert+=1
                self.vertices_decode.append(v_cur)

                self.layer_vert_current.append(v_cur.i)
                self.layer_inlayer_mat_embs.append([])
                self.layer_outlayer_mat_embs.append([])

                i+=1
                continue

        
        vertices=[]
        for v in self.vertices_decode:
            vertices.append(v.undiscrete_vertex(color='cc_face'))

        self.faces=edges_to_faces(self.edges_decode)
        
        
        return vertices, self.faces

    def decode(self, tokens, discrete_bins, mode, colorful, manifix): 
        
        # print('-------------------------start decoding ori----------------------')
        self.init_BO(mode)

        self.edges_decode=[]
        self.vertices_decode=[]
        self.num_vert=0

        self.num_CC_decode=0
        self.CC_decode_inlayer_matrix=[]
        self.CC_decode_outlayer_matrix=[]
        self.CC_decode_vert_queue=[]
        i=0
        # layer up will clean
        self.layer_inlayer_mat_embs=[]
        self.layer_outlayer_mat_embs=[]
        self.layer_vert_current=[]
        self.layer_vert_last=[]
        # new faces decode
        self.decode_all_faces=[]

        self.current_layer=0
        
        while i<len(tokens):
            cur_token=tokens[i]
            if cur_token >= self.num_word_table or cur_token == -1:
                print(f'[WARNING] meet {cur_token}/{self.num_word_table}, stop')
                break
            if cur_token == self.OP.OP_C:
                self.num_CC_decode+=1
                self.CC_decode_inlayer_matrix.append([])
                self.CC_decode_outlayer_matrix.append([])
                self.CC_decode_vert_queue.append([])
                self.current_layer=0
                i+=1
                continue
            if cur_token in [self.OP.OP_U, self.OP.OP_E]:
                
                if self.current_layer==0:
                    cur_layer_vert=self.layer_vert_current
                    self.uplayer_clean()
                    self.layer_vert_last=cur_layer_vert
                    self.current_layer+=1
                    self.CC_decode_inlayer_matrix[self.num_CC_decode-1].append(None)
                    self.CC_decode_outlayer_matrix[self.num_CC_decode-1].append(None)
                    self.CC_decode_vert_queue[self.num_CC_decode-1].append(cur_layer_vert)
                    
                    i+=1
                    continue

                inlayer_M=len(self.layer_inlayer_mat_embs)
                outlayer_M=len(self.layer_outlayer_mat_embs)
                layer_M=len(self.layer_vert_current)
                last_layer_N=len(self.layer_vert_last)
                if layer_M!=inlayer_M or layer_M!=outlayer_M:
                    raise Exception('[Decoding] matrix dim wrong')
                try:
                    inlayer_matrix=self.decompress_inlayer_matrix(self.layer_inlayer_mat_embs)
                    outlayer_matrix=self.decompress_outlayer_matrix(self.layer_outlayer_mat_embs, last_layer_N)
                    edges_in=self.decode_inlayer_matrix(self.layer_vert_current, inlayer_matrix)
                    edges_out=self.decode_outlayer_matrix(self.layer_vert_last, self.layer_vert_current, outlayer_matrix)
                except Exception as e:
                    print(f'[Decode] {str(e)}')
                    break
                self.edges_decode+=edges_in
                self.edges_decode+=edges_out

                cur_layer_vert=self.layer_vert_current
                self.uplayer_clean()
                self.layer_vert_last=cur_layer_vert
                self.current_layer+=1

                self.CC_decode_inlayer_matrix[self.num_CC_decode-1].append(inlayer_matrix)
                self.CC_decode_outlayer_matrix[self.num_CC_decode-1].append(outlayer_matrix)
                self.CC_decode_vert_queue[self.num_CC_decode-1].append(cur_layer_vert)
                # Old version for decoding, may have some slight flipping error in some special topology
                # Updated version is cleaning...
                self.decode_all_faces+=decode_inlayer_connect_faces(inlayer_matrix, cur_layer_vert)
                self.decode_all_faces+=decode_layer_face(self.CC_decode_inlayer_matrix[self.num_CC_decode-1][-2], inlayer_matrix, outlayer_matrix, self.CC_decode_vert_queue[self.num_CC_decode-1][-2], cur_layer_vert)
                i+=1
                continue
            cur_token-=self.OP.OP_NUM
            if cur_token < self.topology_num: # topology token
                target_index=len(self.layer_vert_current)-1
                if cur_token<self.inlayer_emb_num: # this is inlayer emb
                    self.layer_inlayer_mat_embs[target_index].append(cur_token)
                else:
                    self.layer_outlayer_mat_embs[target_index].append(cur_token)
                i+=1
                continue
            else: # normal vertex
                try:
                    v_cur, i = self.decode_next_v(tokens=tokens, i=i, layer=self.current_layer, index=self.num_vert, discrete_bins=discrete_bins, mode=mode)
                except Exception as e:
                    print(f'[Decode] {str(e)}')
                    break
                self.num_vert+=1
                v_cur.decode_CC = self.num_CC_decode # 1, 2, ...
                self.vertices_decode.append(v_cur)

                self.layer_vert_current.append(v_cur.i)
                self.layer_inlayer_mat_embs.append([])
                self.layer_outlayer_mat_embs.append([])

                i+=1
                continue
        
        vertices_layerColor=[]
        vertices_ccColor=[]
        vert_dic_list=[]
        for v in self.vertices_decode:
            vertices_layerColor.append(v.undiscrete_vertex(color='layer'))
            vertices_ccColor.append(v.undiscrete_vertex(color='cc_decode'))
            vert_dic_list.append({'v6_bfs':v.undiscrete_vertex(color='layer'), 'v6_cc': v.undiscrete_vertex(color='cc_decode'),'index': v.i,'layer': v.bfs_layer, 'cc_id': v.decode_CC})
        
        
        # Old version for decoding faces directly based on edges
        # Faster but can not make sure flipping consistency, Abandoned
        # Try this if you have any special need.
        # self.faces=edges_to_faces(self.edges_decode)

        triangles=self.decode_all_faces
        unique_tri_sort=[]
        unique_tri=[]
        for f in triangles:
            f_sort=tuple(np.sort(f))
            if f_sort in unique_tri_sort:
                continue
            unique_tri_sort.append(f_sort)
            unique_tri.append(f)
        triangles=np.array(unique_tri)
        repeat_face=len(self.decode_all_faces)-len(triangles)
        if repeat_face:
            print(f'[Decode] remove {repeat_face} repeat faces')

        if manifix:
            try:
                # The old version of manifold repair and water tight fixing
                vert_dic_list, fixed_faces=manifold_fix(vert_dic_list, triangles)
            except Exception as e:
                fixed_faces=triangles
                print(f'[Manifix] wrong for manifix')

        vertices=None
        ret_face=None
        if colorful:
            vertices=[vertices_layerColor, vertices_ccColor]
        else:
            vertices=vertices_layerColor

        if manifix:
            ret_face=[triangles, fixed_faces]
        else:
            ret_face=triangles
        
        return vertices, ret_face



    def translate_tokens(self, tokens, discrete_bins, mode):
        self.init_BO(mode=mode)
        i=0
        num_vert=0
        translated_tokens=[]
        
        while i<len(tokens):
            this_token=tokens[i]
            token_trans, i=self.translate_token_one(this_token, discrete_bins, mode, tokens, i)
            translated_tokens+=token_trans
            i+=1

        return translated_tokens
    
    def translate_tokens_direct(self, tokens):
        
        i=0
        translated_tokens=[]
        
        while i<len(tokens):

            this_token=tokens[i]
            token_trans=self.translate_token_one_compare(this_token)
            translated_tokens+=token_trans
            i+=1

        return translated_tokens

    def translate_token_one(self, token_one, discrete_bins, mode, tokens, i):
        if token_one < self.OP.OP_NUM:
            if token_one == self.OP.OP_C:
                return ['C'], i
            elif token_one == self.OP.OP_U:
                return ['U'], i
            elif token_one == self.OP.OP_E:
                return ['E'], i
        token_one-=self.OP.OP_NUM
        if token_one<self.topology_num:
            if token_one<self.inlayer_emb_num:
                if token_one<self.layer_vertex_limit:
                    return [f'[S1_{token_one}]'], i
                else:
                    return [f'[S0_{token_one}]'], i
            else:
                Of, Oc=self.from_row_embid(token_one)
                return [f'{{{Of}_{Oc}}}'], i
        else:
            
            v_cur, i = self.decode_next_v(tokens=tokens, i=i, layer=1, index=0, discrete_bins=discrete_bins, mode=mode)
            x,y,z=v_cur.x,v_cur.y,v_cur.z

            return [f'({x},{y},{z})'], i
        
    def translate_token_one_compare(self, token_one):
        if token_one == self.num_word_table:
            return ['EOS']
        elif token_one == -1 or token_one == self.num_word_table+1:
            return ['PAD']
        elif token_one > self.num_word_table+1:
            return ['?']
            
        if token_one < self.OP.OP_NUM:
            if token_one == self.OP.OP_C:
                return ['C']
            elif token_one == self.OP.OP_U:
                return ['U']
            elif token_one == self.OP.OP_E:
                return ['E']
        token_one-=self.OP.OP_NUM
        if token_one<self.topology_num:
            if token_one<self.inlayer_emb_num:
                if token_one<self.layer_vertex_limit:
                    return [f'[S1_{token_one}]']
                else:
                    return [f'[S0_{token_one}]']
            else:
                Of, Oc=self.from_row_embid(token_one)
                return [f'{{{Of}_{Oc}}}']
        else:
            token_one-=self.topology_num

            return [f'(V_{token_one})'] # start from 0
                

    def translate_compare_tokens(self, tokens, tokens_gt):
        
        i=0
        num_vert=0
        translated_tokens=[]
        N_pred=len(tokens)
        N_gt=len(tokens_gt)
    
        if N_pred<N_gt:
            tokens_longer=tokens_gt
            longger_name='gt'
            N_min=N_pred
            N_max=N_gt
        else:
            tokens_longer=tokens
            longger_name='pred'
            N_min=N_gt
            N_max=N_pred
        
        while i<N_min:
            pred_token=tokens[i]
            gt_token=tokens_gt[i]
            if pred_token == gt_token:
                gt_trans=self.translate_token_one_compare(gt_token)
                translated_tokens.append(f'#={gt_trans[0]}')
                i+=1
                continue
            gt_trans=self.translate_token_one_compare(gt_token)
            pred_trans=self.translate_token_one_compare(pred_token)
            translated_tokens.append(f'{gt_trans[0]}--{pred_trans[0]}')
            i+=1

        translated_tokens.append(f'\n {longger_name} is longer \n')
        while i<N_max:
            this_token=tokens_longer[i]
            this_trans=self.translate_token_one_compare(this_token)
            translated_tokens.append(this_trans[0])
            i+=1

        return translated_tokens

def generate_symmetric_random_matrix_with_ones_on_diagonal(N):

    upper_tri = np.triu(np.random.randint(0, 2, size=(N, N)), k=1)

    np.fill_diagonal(upper_tri, 1)

    symmetric_matrix = np.maximum(upper_tri, upper_tri.T)
    return symmetric_matrix

def generate_random_matrix(M, N):

    matrix = np.random.randint(2, size=(M, N))
    return matrix

if __name__ == "__main__":
    
    engine=Engine_SilkSong(discrete_bins=128)
    import ipdb
    from tqdm import tqdm
    print(f'inlayer0/inlayer1/outlayer/all: {engine.layer_vertex_limit}/{engine.inlayer_emb_num-engine.layer_vertex_limit}/{engine.outlayer_emb_num}/{engine.topology_num}')
    
    # debug outlayer
    # engine.set_outlayer_window(5)
    # print(engine.all_situation_num)
    # for n in tqdm(range(2, 200)):
    #     for i in range(100):
    #         outlayer=generate_random_matrix(n,30)
    #         # print(inlayer_t)
    #         emb_i, N=engine.compress_outlayer_matrix(outlayer)
    #         # print(comp_id)
    #         outlayer_d=engine.decompress_outlayer_matrix(emb_i, N)
    #         # print(inlayer_d)
    #         if not np.array_equal(outlayer, outlayer_d):
                
    #             print(f'n: {n}')
    #             ipdb.set_trace()


    # debug inlayer
    # for n in tqdm(range(2, 256)):
    #     for i in range(100):
    #         inlayer_t=generate_symmetric_random_matrix_with_ones_on_diagonal(n)
    #         # print(inlayer_t)
    #         comp_id=engine.compress_inlayer_matrix(inlayer_t)
    #         # print(comp_id)
    #         inlayer_d=engine.decompress_inlayer_matrix(comp_id)
    #         # print(inlayer_d)
    #         if not np.array_equal(inlayer_t, inlayer_d):
    #             delta=inlayer_t-inlayer_d
    #             print(f'n: {n}')
    #             ipdb.set_trace()
    # inlayer_t=generate_symmetric_random_matrix_with_ones_on_diagonal(18)
    # print(inlayer_t)
    # comp_id=engine.compress_inlayer_matrix(inlayer_t)
    # print(comp_id)
    # inlayer_d=engine.decompress_inlayer_matrix(comp_id)
    # print(inlayer_d)
    # if not np.array_equal(inlayer_t, inlayer_d):
    #     delta=inlayer_t-inlayer_d
    #     ipdb.set_trace()