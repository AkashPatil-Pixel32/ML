## Kmeans clustering example

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import numpy as np

def kmeans_cluster(df, num_clusters = n):
    kmeans = KMeans(n_clusters = n, init = 'random')
    model = kmeans.fit(df)
    op_kmeans = model.predict(df)
    c = kmeans.cluster_centers_
    return op_kmeans, c
    
if __name__=="__main__": 
    df = pd.read_csv('data.csv')
    op_kmeans, cluster = kmeans_cluster(df)
    plt.scatter(df['x0'], df['x1'], c = op_kmeans, s=50, cmap='Set1')
    plt.grid()
    plt.scatter(c[:,0], c[:,1], c='black', s=200)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.matplotlib.patches.CirclePolygon((c[0][0], c[0][1]), radius=7, resolution=20, color='r')
    
    c0_c = c[0]
    c1_c = c[1]
    cluster1_p = df[op_kmeans==0]
    cluster2_p = df[op_kmeans==1]

    c1_d = cluster1_p.apply(lambda x:np.sqrt((x[0]-c0_c[0])**2+(x[1]-c0_c[1])**2),axis=1)
    c2_d = cluster2_p.apply(lambda x:np.sqrt((x[0]-c1_c[0])**2+(x[1]-c1_c[1])**2),axis=1)
    print(c1_d.mean())
    print(c2_d.mean())