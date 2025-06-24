import numpy as np

class UnionFind:  
    def __init__(self, size):  
        self.parent = list(range(size))  
        self.rank = [1] * size  

    def find(self, a):  
        if self.parent[a] != a:  
            self.parent[a] = self.find(self.parent[a])  # Path compression  
        return self.parent[a]  

    def union(self, a, b):  
        rootA = self.find(a)  
        rootB = self.find(b)  

        if rootA != rootB:  
            # Union by rank  
            if self.rank[rootA] > self.rank[rootB]:  
                self.parent[rootB] = rootA  
            elif self.rank[rootA] < self.rank[rootB]:  
                self.parent[rootA] = rootB  
            else:  
                self.parent[rootB] = rootA  
                self.rank[rootA] += 1 
    
    def extend(self):
        cur_size=len(self.parent)
        self.parent.append(cur_size)
        self.rank.append(1)
        return cur_size
    
    def group_num(self, cc_list):
        root_list=[self.find(x) for x in cc_list]
        return len(set(root_list))
    
    def union_group(self, cc_list):
        for i in range(len(cc_list)-1):
            a=cc_list[i]
            b=cc_list[i+1]
            self.union(a, b)
        return self.find(cc_list[0])
    
    def parent_list(self):
        return self.parent, len(set(self.parent))
    

def edge_key(a, b):  
    return (a, b) if a < b else (b, a)

def check_flip(f, FN):
    edge_right=[]
    edge_fix=[]
    next_neighbors=[]
    for j in range(FN):  
          
        if f.half_edges[j].o is None:  
            # this is bound
            edge_right.append(True)
            edge_fix.append(False)
            continue
        else:
            this_he = f.half_edges[j]
            edge_fix.append(this_he.o.t.fix_orientation)
            if not this_he.o.t.fix_orientation:
                next_neighbors.append(this_he.o.t.i)
            if this_he.s.i==this_he.o.s.i and this_he.e.i==this_he.o.e.i:
                edge_right.append(False)
            else:
                edge_right.append(True)
    if sum(edge_right)==3: # if we need flip / next_neighbor_faces flip / flip process success?
        return False, next_neighbors, True
    else:
        if sum(edge_fix) ==0:
            # print(f'find free face {f.i}')
            if sum(edge_right)<2:
                f.flip()
                return True, next_neighbors, True
            else:
                return False, next_neighbors, True
        # 1 or 2 or 3 neighbor faces fixed
        flag_flip = None
        flag_notflip = None
        for index_edge_fix in range(3):
                    
            # if edge_fix[index_edge_fix] == True and edge_right[index_edge_fix] == False:
            #     flag_flip = True
            # elif edge_fix[index_edge_fix] == True and edge_right[index_edge_fix] == True:
            #     flag_notflip = True

            if edge_fix[index_edge_fix] is True:
                if edge_right[index_edge_fix] is False:
                    flag_flip=True
                else:
                    flag_notflip=True

        if flag_flip and flag_notflip:
            # ipdb.set_trace()
            # raise Exception('[E] face flip wrong')
            return False, next_neighbors, False
        if flag_flip:
            f.flip()
            return True, next_neighbors, True
        elif flag_notflip:
            return False, next_neighbors, True
        else:
            raise Exception('[E] face flip wrong 2')
        

def triangle_area_3d(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    v1 = p2 - p1  
    v2 = p3 - p1  
    
    cross_product = np.cross(v1, v2)
    area = np.linalg.norm(cross_product) / 2
    
    return area