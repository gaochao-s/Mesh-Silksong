from math import comb

def combination_to_index(combination, n):
    """
    给定组合，返回其编号。
    
    :param combination: 包含k个元素的列表，表示组合。
    :param n: 元素取值范围是1到n。
    :return: 组合的编号。
    """
    k = len(combination)
    index = 0
    for i in range(k):
        element = combination[i]
        if i > 0:
            prev_element = combination[i - 1]
        else:
            prev_element = 0
        
        for j in range(prev_element + 1, element):
            index += comb(n - j, k - i - 1)
    
    return index

def index_to_combination(index, k, n):
    """
    给定编号，返回组合。
    
    :param index: 组合的编号。
    :param k: 组合中元素的个数。
    :param n: 元素取值范围是1到n。
    :return: 对应的组合。
    """
    combination = []
    current_index = index
    start = 1
    
    for i in range(k):
        for j in range(start, n + 1):
            count = comb(n - j, k - i - 1)
            if current_index < count:
                combination.append(j)
                start = j + 1
                break
            current_index -= count
    
    return combination

def generate_combination_mappings(n, k):
    """
    生成组合与编号的映射关系。
    
    :param n: 元素取值范围是1到n。
    :param k: 组合中元素的个数。
    :return: (组合到编号的字典, 编号到组合的列表)
    """
    def generate_combinations(start, k, n, current_combination, all_combinations):
        if k == 0:
            all_combinations.append(list(current_combination))
            return
        for i in range(start, n + 1):
            current_combination.append(i)
            generate_combinations(i + 1, k - 1, n, current_combination, all_combinations)
            current_combination.pop()

    all_combinations = []
    generate_combinations(1, k, n, [], all_combinations)

    combination_to_index_map = {}
    index_to_combination_map = []

    for combination in all_combinations:
        index = combination_to_index(combination, n)
        combination_to_index_map[tuple(combination)] = index
        index_to_combination_map.append(combination)

    return combination_to_index_map, index_to_combination_map
