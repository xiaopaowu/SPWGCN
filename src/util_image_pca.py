'''
Author: Wuyifan
Date: 2023-10-14 12:45:22
LastEditors: Wuyifan
LastEditTime: 2024-02-27 19:53:46
'''
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as scio


project_list =["QuXiangGIF","Expression","HuaWei","YZOffice","XinChuangOS",'GuDong',"KunPeng"]
project_size = []  

# def chi2_distance(histA, histB, eps=1e-10):
#     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
#     return d
def load_data(project_name):
    path = 'data/pyramids_feature/pyramids_' + project_name + '_200_3.mat'

    data = scio.loadmat(path)
    feature_vector = np.array(data['pyramid_all'])

    text_filename = 'data/image_feature_vector/' + project_name + '.txt'
    np.savetxt(text_filename, feature_vector, fmt='%.6f', delimiter=' ', encoding='utf-8')


X = np.array([[0] * 4200])
for project in project_list:
    load_data(project)
    filename = 'data/image_feature_vector/' + project + '.txt'
    tmp = np.loadtxt(filename)

    X = np.append(X, tmp, axis=0)
    project_size.append(tmp.shape[0])
    # print(X, '\n')

X = X[1:]
# print(X.shape)

pca = PCA(300)  # int: 300（300个特征）, float: 0.95（保留95%的信息）, str: 'mle'（自动选择特征个数）
pca.fit(X)

# 输出特征值
# print(pca.explained_variance_ratio_)
# 输出特征向量
# print(pca.components_)

# 降维后的数据

X_new = pca.transform(X)
print('The image feature after pca processing: ', X_new.shape)
print(X_new)
print(max(X_new.reshape(X_new.shape[0] * X_new.shape[1], 1)), '\n')

# fig = plt.figure()
# plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
# plt.show()

for i in range(1, 7):
    project_size[i] += project_size[i - 1]
print('The size of each project: ', project_size, '\n')

m, n = X_new.shape
X_new = preprocessing.minmax_scale(X_new.flatten(), feature_range=(0, 1)).reshape(m, n)
print('The image feature after minmax scaling: ', X_new.shape)

img = [[] for i in range(7)]
img[0] = X_new[:project_size[0]]
img[1] = X_new[project_size[0]:project_size[1]]
img[2] = X_new[project_size[1]:project_size[2]]
img[3] = X_new[project_size[2]:project_size[3]]
img[4] = X_new[project_size[3]:project_size[4]]
img[5] = X_new[project_size[4]:project_size[5]]
img[6] = X_new[project_size[5]:]

print(X_new)

for index, project in enumerate(project_list):
    path = 'data/image_vector/' + project + '.txt'

    img_vector = np.array(img[index])
    np.savetxt(path, img_vector, fmt="%.6f", delimiter=" ")
