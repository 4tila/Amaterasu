import numpy as np
path = './path/'
noise = True
prefix = "stochastic" if noise else "deterministic"
dimension = 170 if noise else 130
S = np.load(f'{path}{prefix}_PCA_S.npy')
VT = np.load(f'{path}{prefix}_PCA_VT.npy')
projection_matrix = VT.T[:, :dimension]/S[:dimension]
np.save(f'{path}{prefix}_projection_matrix.npy', projection_matrix)

curvature = np.concatenate([np.abs(np.diff(np.log(S), 2)), [0, 0]]) # making last two entries zero just to match dimensions
S = np.array([S, curvature]).T
np.savetxt(f'{path}{prefix}_S.txt', S, delimiter=' ', fmt='%16.6E', header='Singular Values') # saving for plotting later
