import numpy as np
from math import erf
def p(z): return 1 - erf(abs(z)*0.7071067811865475)
def compare(row,table, i, j, N=5000):
    p1, p2 = table[row][i]*1e-2, table[row][j]*1e-2
    SE1 = np.sqrt(p1*(1-p1)/N)
    SE2 = np.sqrt(p2*(1-p2)/N)
    SE = np.linalg.norm([SE1, SE2])
    return abs(p1-p2)/SE
ablation = ["non-ablated", "substitution by mean ablation", "substitution by zero ablation", "shuffle ablation"]
table = np.array([  [22.08, 20.06, 19.34, 18.54],\
                    [29.92, 27.38, 26.72, 25.88],\
                    [17.14, 16.60, 16.60, 15.96],\
                    [24.78, 24.00, 23.90, 23.32]])
print(table)
comparisons = [(0, 1), (1, 2), (0, 3)]
print('_'*150)
for (x, y) in comparisons:
    for row in range(4):
        C = compare(row, table, x, y)
        print(f"""Z-score between accuracy of {"deterministic" if row <2 else "stochastic"} model for {ablation[x]} and {ablation[y]} for {"top-10" if bool(row&1) else "top-1"} predictions: {round(C, 3)} with probability of observing a value at least this extreme of {round(100*p(C), 4)}%\n""")
    print('_'*150)
