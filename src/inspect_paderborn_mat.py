from scipy.io import loadmat

mat = loadmat("../data/paderborn/K001/N09_M07_F10_K001_1.mat")
mat = {k: v for k, v in mat.items() if not k.startswith("__")}

data = list(mat.values())[0]
X = data['X'][0, 0]

print(X.dtype)

for i in range(X['Name'].shape[1]):
    print(i, X['Name'][0, i])
