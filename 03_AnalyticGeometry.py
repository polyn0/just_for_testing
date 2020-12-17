## 3. Analytic Geometry


# Manhattan Norm( l1 norm)
# Euclidean Norm( l2 norm)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams[ "figure.figsize" ] = (10,10) # plot ������ ����

# 1. ��2norm �׸���
xRight = np.linspace(0,1,50) # 0 ~ 1
xLeft = np.linspace(-1,0,50) # -1 ~ 0

xarr = [xRight, xLeft, xLeft, xRight] # x coordinate for Q1, Q2, Q3, Q4
xarr = np.array(xarr) # list -> array (4, 50)
xarr = xarr.reshape(-1) # �� �ٷ� (200)

yarr = [np.sqrt(1-xRight**2), np.sqrt(1-xLeft**2), -np.sqrt(1-xLeft**2), -np.sqrt(1-xRight**2)]
yarr = np.array(yarr).reshape(-1)

plt.scatter(xarr, yarr, s = 5, color = 'r')


# 2. l1norm �׸���
xarr = [xRight, xLeft, xLeft, xRight] # x coordinate for Q1, Q2, Q3, Q4
xarr = np.array(xarr) # list -> array (4, 50)
xarr = xarr.reshape(-1) # �� �ٷ� (200)

yarr = [1-xRight, 1+xLeft, -(1+xLeft), -(1-xRight)]
yarr = np.array(yarr).reshape(-1)

plt.scatter(xarr, yarr, s = 5, color = 'b')


plt.title('Manhattan NOrm(L1, blue), Euclidean Norm(L2, red)')
plt.legend(["l2", "l1"])
plt.show()