import numpy as np

def gen_split(y1, y2):
    # 计算导数数组
    x = range(len(y1))
    derta_y1 = np.gradient(y1,x)
    derta_y2 = np.gradient(y2,x)

    # 贪心策略
    greedy_split = -1
    greedy_val = -1
    for i in range(len(x)):
        gi_l = 0
        gi_r = 0
        for j in range(0, i):
            gi_l += (derta_y1[j]-derta_y2[j]) / abs(derta_y1[j]-derta_y2[j])
        for j in range(i+1, len(x)):
            gi_r += (derta_y1[j]-derta_y2[j]) / abs(derta_y1[j]-derta_y2[j])
        
        gi_l, gi_r = abs(gi_l), abs(gi_r)
        print(gi_l, gi_r)
       
        if (gi_l + gi_r) >= greedy_val:
            greedy_val = gi_l + gi_r
            greedy_split = i
    
    return greedy_split

    
A = [0.07815, 0.07667, 0.07350, 0.06968, 0.06457, 0.05809, 0.04816, 0.03223, 0.01705, 0.01005]
P = [0.06389, 0.05617, 0.04806, 0.04096, 0.03598, 0.03068, 0.02545, 0.01936, 0.01398, 0.01005]

H = [0.06283, 0.06127, 0.06015, 0.05640 , 0.05124, 0.04315, 0.03288, 0.02215, 0.01363, 0.01005]
L = [0.07557, 0.06747, 0.05893, 0.04829, 0.03841, 0.02830 , 0.02048, 0.01473, 0.01143, 0.01005]

# A = [0.06915, 0.06806, 0.06690 , 0.06477, 0.06126, 0.05669, 0.04822, 0.03384, 0.01743, 0.01013]
# P = [0.03492, 0.03365, 0.03238, 0.03078, 0.02907, 0.02708, 0.02403, 0.01894, 0.01379, 0.01013]

# H = [0.05797, 0.05728, 0.05636, 0.05414 , 0.05, 0.04295, 0.03245, 0.02157, 0.01347, 0.01013]
# L = [0.0281, 0.02698, 0.02522, 0.02302, 0.01944, 0.01599 , 0.01311, 0.01135, 0.01039, 0.01013]
print(f"AP timestep split is {gen_split(y1=A, y2=P)}")
print(f"HL timestep split is {gen_split(y1=H, y2=L)}")