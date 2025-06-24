import numpy as np
import networkx as nx

def edge_key(a, b):  
    return (a, b) if a < b else (b, a)

def get_edge_to_face_dic(face_list, triangles, vert_index):
    ret_dic={}
    for face_index in face_list:
        edge=[x for x in triangles[face_index] if x !=vert_index]
        ret_dic[edge_key(edge[0], edge[1])]=face_index
    return ret_dic

def find_cycle_or_chain(component):  
    # component: subgraph
    if not nx.is_tree(component):  
        # 查找所有的简单环  
        cycles = list(nx.simple_cycles(component.to_directed()))  
        if cycles:  
            max_cycle = max(cycles, key=len)  
            if max_cycle == 2:
                raise Exception('wrong')
            return max_cycle, 'cycle'  
            # max_cycle = min(cycles, key=len)  
            # return max_cycle, 'cycle'  
        raise Exception('can not find cycle ?')
    
    # 如果没有环，返回最长路径  
    longest_path = []  
    for node in component.nodes():  
        # 进行深度优先搜索找最长路径  
        paths = nx.single_source_dijkstra_path(component, node)  
        for path in paths.values():  
            if len(path) > len(longest_path):  
                longest_path = path  
    return longest_path, 'chain'

def find_all_cycle(component):  
    # component: subgraph
    if not nx.is_tree(component):  
        # 查找所有的简单环  
        cycles = list(nx.simple_cycles(component))  

        if cycles:  
            return cycles, 'cycle'
        raise Exception('can not find cycle ?')
     
    longest_path = []  
    return longest_path, 'chain'

class VertDecode:
    def __init__(self, v6_bfs, v6_cc, index, layer, cc_id):
        self.v6_bfs=v6_bfs
        self.v6_cc=v6_cc
        self.index=index
        self.cc_id=cc_id
        self.neighbor_faces=[]


class MeshDecode:  
    def __init__(self, vertices_list, faces): 
        self.verts=[VertDecode(ele['v6_bfs'], ele['v6_cc'], ele['index'], ele['layer'], ele['cc_id']) for ele in vertices_list]
        self.faces=faces
        self.vert_num=len(self.verts)
        self.face_num=len(faces)
        self.face_exclude=[]
        self.bound_edges=[]
        self.hole_fix_faces=[]

    def exclude_nm_faces(self):
        for face_index, face in enumerate(self.faces):
            self.verts[face[0]].neighbor_faces.append(face_index)
            self.verts[face[1]].neighbor_faces.append(face_index)
            self.verts[face[2]].neighbor_faces.append(face_index)

        for vert_index in range(self.vert_num):
            face_list_o=self.verts[vert_index].neighbor_faces
            face_list=[face for face in face_list_o if face not in self.face_exclude]
            
            if not face_list:
                print(f'[WARNING] vert {vert_index} single')
                continue
            edge_to_face_dic=get_edge_to_face_dic(face_list, self.faces, vert_index)
            # 只要有环，排除环外的全部face，全部连通分量，且无边界边
            # 若无环，对于每个连通分量，计算最长链条，排除非流形face，标记边界边
            # 修复一般孔洞：cycle 长度< 6 且同一CC
            # 如果要水密性修复，则xxx
            info_dic=self.check_edge_graph(vert_index, edge_to_face_dic)
            keeped_face=info_dic['keeped_face']
            exclude_faces=[f for f in face_list if f not in keeped_face]
            self.face_exclude+=exclude_faces
            if info_dic['type']=='chain':
                self.bound_edges+=info_dic['bound_edges']

        print(f'about {len(self.face_exclude)} excluded')
        print(f'find {len(self.bound_edges)} bound edges')
    
    def fix_hole(self, max_cycle=6, water_tight=False):
        if water_tight:
            max_cycle=100
        hole_graph=nx.Graph()
        hole_graph.add_edges_from(self.bound_edges)
        components =  list(nx.connected_components(hole_graph))
        for cc_i, component_nodes in enumerate(components):
            component = hole_graph.subgraph(component_nodes)
            cycles, type = find_all_cycle(component)
            if type=='chain':
                continue
            for cycle in cycles:
                cycle_len=len(cycle)
                if cycle_len>max_cycle:
                    continue
                # fix hole
                self.hole_fix_faces+=self.fix_cycle(cycle)
        print(f'fix hole and add {len(self.hole_fix_faces)} faces')

    def fix_cycle(self, cycle):
        if len(cycle)==3:
            a, b, c= cycle
            return [[a,b,c]]
        if len(cycle)==4:
            a,b,c,d=cycle
            return [[a,d,b],[c,b,d]]
        if len(cycle)==5:
            a,b,c,d,e=cycle
            return [[a,c,b],[a,d,c],[a,e,d]]
        if len(cycle)==6:
            a,b,c,d,e,f=cycle
            return [[a,c,b],[a,d,c],[a,f,d],[f,e,d]]
        raise Exception('not implement')
        

    
    def check_edge_graph(self, vert_index, edge_to_face_dic):
        graph=nx.Graph()
        graph.add_edges_from(edge_to_face_dic.keys())
        components=list(nx.connected_components(graph))
        cc_info_list=[]
        cycle_cc_index=[]
        for cc_i, component_nodes in enumerate(components):
            component = graph.subgraph(component_nodes)
            cycle_or_chain, type = find_cycle_or_chain(component)
            cc_dic={}
            cc_dic['type']=type
            if type == 'cycle':
                
                cc_edges_cycle=[edge_key(cycle_or_chain[i%len(cycle_or_chain)], cycle_or_chain[(i+1)%len(cycle_or_chain)]) for i in range(len(cycle_or_chain))]
                cycle_len=len(cc_edges_cycle)
                cc_dic['cycle_edge']=cc_edges_cycle
                cc_dic['cycle_face']=[edge_to_face_dic[edge] for edge in cc_edges_cycle]
                cc_dic['cycle_len']=cycle_len
                if not cycle_cc_index:
                    cycle_cc_index.append([cc_i, cycle_len])
                else:
                    last_len=cycle_cc_index[0][1]
                    if cycle_len>last_len:
                        cycle_cc_index=[[cc_i, cycle_len]]
            else:
                cc_edges_chain=[edge_key(cycle_or_chain[i], cycle_or_chain[(i+1)]) for i in range(len(cycle_or_chain)-1)]
                cc_dic['chain_edge']=cc_edges_chain
                cc_dic['chain_face']=[edge_to_face_dic[edge] for edge in cc_edges_chain]
                cc_dic['chain_len']=len(cc_edges_chain)
                cc_dic['chain_bound']=[cycle_or_chain[0], cycle_or_chain[-1]]

            cc_info_list.append(cc_dic)
        
        ret_dic={}
        if cycle_cc_index:
            # 有cycle且找到最大cycle, 则保留这些cycle的face，去除其它face
            cc_i, cycle_len = cycle_cc_index[0]
            keeped_face=cc_info_list[cc_i]['cycle_face']
            ret_dic['type']='cycle'
            ret_dic['keeped_face']=keeped_face
        else:
            # 全是chain
            keeped_face=[]
            bound_edges=[]
            for chain_dic in cc_info_list:
                keeped_face+=chain_dic['chain_face']
                bp_1=chain_dic['chain_bound'][0]
                bp_2=chain_dic['chain_bound'][1]
                bound_edges.append([vert_index, bp_1])
                bound_edges.append([vert_index, bp_2])
            ret_dic['type']='chain'
            ret_dic['keeped_face']=keeped_face
            ret_dic['bound_edges']=bound_edges
        return ret_dic
    
        

def manifold_fix(vertices_dic_list, faces):
    # vertices: list
    # faces: list
    mesh_input=MeshDecode(vertices_list=vertices_dic_list, faces=faces)
    mesh_input.exclude_nm_faces()
    mesh_input.fix_hole()
    new_faces=[f for idx, f in enumerate(mesh_input.faces) if idx not in mesh_input.face_exclude]
    new_faces+=mesh_input.hole_fix_faces

    return vertices_dic_list, new_faces

if __name__ == "__main__":
    pass