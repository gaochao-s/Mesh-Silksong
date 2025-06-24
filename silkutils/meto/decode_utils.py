import numpy as np
import networkx as nx



def decode_layer_face(inlayer_matlast, inlayer_matcur, outlayer_mat, last_verts, cur_verts):
    M, N = outlayer_mat.shape
    face_all=[]
    for i in range(M): # check inlayer mat last
        up_v=cur_verts[i]
        down_index_list=[]
        for j in range(N):
            if outlayer_mat[i][j]==1:
                down_index_list.append(j)
        N_down=len(down_index_list)
        if N_down == 1:
            continue
        if N_down == 2:
            ind_1=down_index_list[0]
            ind_2=down_index_list[1]
            if abs((ind_2-ind_1)%N) > abs((ind_1-ind_2)%N):
                down_index_1=ind_2
                down_index_2=ind_1
            else:
                down_index_1=ind_1
                down_index_2=ind_2
            if check_inlayer(down_index_1, down_index_2, inlayer_matlast):
                face_all.append([up_v, last_verts[down_index_2], last_verts[down_index_1]])
            continue
        for k in range(N_down):
            down_index_1=down_index_list[k%N_down]
            down_index_2=down_index_list[(k+1)%N_down]
            if check_inlayer(down_index_1, down_index_2, inlayer_matlast):
                face_all.append([up_v, last_verts[down_index_2], last_verts[down_index_1]])
    
    for j in range(N):
        down_v=last_verts[j]
        up_index_list=[]
        for i in range(M):
            if outlayer_mat[i][j]==1:
                up_index_list.append(i)
        N_up=len(up_index_list)
        if N_up == 1:
            continue
        if N_up == 2:
            ind_1=up_index_list[0]
            ind_2=up_index_list[1]
            if abs((ind_2-ind_1)%M) > abs((ind_1-ind_2)%M):
                up_index_1=ind_2
                up_index_2=ind_1
            else:
                up_index_1=ind_1
                up_index_2=ind_2
            if check_inlayer(up_index_1, up_index_2, inlayer_matcur):
                face_all.append([down_v, cur_verts[up_index_1], cur_verts[up_index_2]])
            continue
        for k in range(N_up):
            up_index_1=up_index_list[k%N_up]
            up_index_2=up_index_list[(k+1)%N_up]
            if check_inlayer(up_index_1, up_index_2, inlayer_matcur):
                face_all.append([down_v, cur_verts[up_index_1], cur_verts[up_index_2]])

    return face_all


def decode_inlayer_connect_faces(inlayer_matrix, vertex_list):
    G = nx.Graph()

    all_edges=[]
    M=len(inlayer_matrix)
    for i in range(M):
        for j in range(i+1, M):
            if inlayer_matrix[i][j]==1:
                all_edges.append((i, j))

    G.add_edges_from(all_edges)
    triangles_index = [tuple(sorted(cycle)) for cycle in nx.cycle_basis(G) if len(cycle) == 3]
    triangles_index_fix=[]
    for triangle_i in triangles_index:
        a, b, c= triangle_i
        triangles_index_fix.append([a, c, b])
    triangles = [[vertex_list[triangle[0]], vertex_list[triangle[1]], vertex_list[triangle[2]]] for triangle in triangles_index_fix]
    return triangles
        

def check_inlayer(down_index_1, down_index_2, inlayer_mat):
    if inlayer_mat is None:
        return True
    if inlayer_mat[down_index_1][down_index_2]==1:
        return True
    else:
        return False
        
def save_matrix_to_txt(mat_in, mat_out, file_name="matrix.txt"):
    with open(file_name, "w") as f:
        CC_num=len(mat_in)
        for i in range(CC_num):
            in_list=mat_in[i]
            out_list=mat_out[i]
            f.write(f"CC {i+1}: \n")
            layer_num=len(in_list)
            for j in range(1, layer_num):
                f.write(f"Matrix layer: {j}\n")
                matrix_in=in_list[j]
                matrix_out=out_list[j]
                matrix_in_str = "\n".join("    " + " ".join(f"{val}" for val in row) for row in matrix_in)
                matrix_out_str = "\n".join("    " + " ".join(f"{val}" for val in row) for row in matrix_out)
                f.write(matrix_in_str + "\n\n") 
                f.write(matrix_out_str + "\n\n")  
            f.write('-------------------\n')