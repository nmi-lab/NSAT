import numpy as np


if __name__ == '__main__':
    rank, size_x, size_y = 7, 70, 60
    w = np.load('adjacent.npy')
    num_units, num_states = w.shape[0], w.shape[2]
    states = [2]
    # layers = [range(2), range(2, 36), range(36, 36+18), range(36+18, num_units)]
    # layers = [range(2, 38), range(38, 38+18), range(38+18, num_units)]
    layers = [range(2), range(38, 38+18), range(38+18, num_units)]
    with open('graph.gv', 'w') as f:
        f.write("digraph G {\n")
        f.write("   ranksep={}; size = \"{}, {}\";\n".format(rank,
                                                             size_x,
                                                             size_y))

        for i in range(len(layers)):
            f.write("{ rank = same; ")
            for j in layers[i]:
                f.write(" {} ".format(j))
            f.write(" }\n")

        for i in range(num_units):
            for j in range(num_units):
                for k in states:
                    if w[i, j, k] == 1:
                        f.write("   "+str(i)+" -> "+str(j)+"; \n")
        f.write("}\n")
