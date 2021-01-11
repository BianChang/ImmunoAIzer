import numpy as np
from numba import jit

@jit
def nuclei_process(pred):
    """define which type of cell """
    pred1 = pred.copy()
    row = pred.shape[0]
    col = pred.shape[1]
    for m in range(row):
        for n in range(col):
            if pred[m][n] == 3:
                for p in [-1, 0, 1]:
                    for q in [-1, 0, 1]:
                        if 0 <= m + p <= row - 1 and 0 <= n + q <= col - 1 and pred[m + p][n + q] == 2:
                            x0 = m
                            y0 = n
                            x1 = m + p
                            y1 = n + q
                            zhan = np.zeros((row * col, 2))
                            zhan_nuclei = np.zeros((row * col, 2))
                            zhan_lcell = np.zeros((row * col, 2))
                            count = 1
                            nuclei_count = 1
                            lcell_count = 1
                            pred[x0][y0] = 0
                            pred[x1][y1] = 0
                            zhan_lcell[lcell_count - 1][0] = x0
                            zhan_lcell[lcell_count - 1][1] = y0
                            zhan_nuclei[nuclei_count - 1][0] = x1
                            zhan_nuclei[nuclei_count - 1][1] = y1
                            zhan[count - 1][0] = x1
                            zhan[count - 1][1] = y1
                            while count > 0:
                                x1 = int(zhan[count - 1][0])
                                y1 = int(zhan[count - 1][1])
                                count = count - 1
                                for i in [-1, 0, 1]:
                                    for j in [-1, 0, 1]:
                                        if 0 <= x1 + i <= row - 1 and 0 <= y1 + j <= col - 1 and pred[x1 + i][
                                            y1 + j] == 2:
                                            x2 = x1 + i
                                            y2 = y1 + j
                                            pred[x2][y2] = 0
                                            nuclei_count = nuclei_count + 1
                                            zhan_nuclei[nuclei_count - 1][0] = x2
                                            zhan_nuclei[nuclei_count - 1][1] = y2
                                            count = count + 1
                                            zhan[count - 1][0] = x2
                                            zhan[count - 1][1] = y2

                                        if 0 <= x1 + i <= row - 1 and 0 <= y1 + j <= col - 1 and pred[x1 + i][
                                            y1 + j] == 3:
                                            x2 = x1 + i
                                            y2 = y1 + j
                                            pred[x2][y2] = 0
                                            lcell_count = lcell_count + 1
                                            zhan_lcell[lcell_count - 1][0] = x2
                                            zhan_lcell[lcell_count - 1][1] = y2
                                            count = count + 1
                                            zhan[count - 1][0] = x2
                                            zhan[count - 1][1] = y2
                            if nuclei_count >= lcell_count:
                                for ind in range(lcell_count):
                                    pred1[int(zhan_lcell[ind][0])][int(zhan_lcell[ind][1])] = 2
                            else:
                                for ind in range(nuclei_count):
                                    pred1[int(zhan_nuclei[ind][0])][int(zhan_nuclei[ind][1])] = 3
    return pred1
