#Array structure for songs
#There will be two arrays of similar shape (108x1000), one for the ground truth notes and the other for the observations (that will include noise)

import numpy as np

zeros=np.zeros((108,1000), dtype=float)

for n in range(len(zeros)):
    arr=np.random.normal(1,0.5,1000)
    for i in arr:
        if i<0.5:
            arr[np.where(arr==i)]=0
        else:
            arr[np.where(arr==i)]=1
        zeros[n]=arr