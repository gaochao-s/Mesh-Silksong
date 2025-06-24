
def get_layer_rgb_color(index):
    
    colors =[
        (255, 0, 0), # r
        (0, 255, 0), # g
        (0, 0, 255), # b
        (255, 255, 255), # white
        (0, 0, 0),  # black
    ]

    if index==0:
        return colors[3]
    return colors[(index - 1) % 3]

def get_distinct_rgb_color(index):
    if not index:
        return (0, 0, 0)
    if index=='B':
        return (0, 0, 0)
    if index=='W':
        return (255,255,255)
    if index=='bl':
        return (135, 206, 250)
    if index=='R':
        return (255, 0, 0)
    colors = [
        (135, 206, 250),  # 浅天蓝
        (144, 238, 144),  # 浅绿色
        (255, 182, 193),  # 浅粉色
        (210, 180, 140),   # 黄褐色
        (218, 112, 214),  # 兰花紫
        (255, 215, 0),    # 金色
        (255, 99, 71),    # 番茄红
        (240, 128, 128),  # 浅珊瑚色
        (173, 216, 230),  # 淡蓝色
        (152, 251, 152),  # 苍绿色
        
    ]

    return colors[(index - 1) % len(colors)]