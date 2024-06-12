import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def top_sort(scores, n=10):
    return np.argsort(-scores)[:n]

def Avg(group):
    return np.mean(group, axis=0)

def AU(group):
    return np.sum(group, axis=0)

def SC(group):
    return np.sum(np.where(group>0,1,0),axis=0)

def AV(group):
    threshold =4
    return np.sum(np.where(group>=threshold,group,0),axis=0)

def BC(group):
    data = pd.DataFrame(group)
    data = data.rank(axis=1)-1
    data = data.sum(axis=0)
    return data.values

def CR(group):
    movie_num = group.shape[1]
    result = np.zeros(movie_num)
    for i in range(movie_num):
        for j in range(i+1, movie_num):
            temp = np.sum(np.where(group[:, i] > group[:, j], 1, np.where(group[:, i] == group[:, j], 0, -1)))
            result[i]+= np.where(temp>0,1,np.where(temp==0,0,-1))
            temp = np.sum(np.where(group[:, j] > group[:, i], 1, np.where(group[:, j] == group[:, i], 0, -1)))
            result[j]+= np.where(temp>0,1,np.where(temp==0,0,-1))
    return result


ratings = pd.read_csv('ratings.dat', sep='::', engine='python', names=['User', 'Movie', 'Rate','time'])

users = ratings['User'].max()
movies = ratings['Movie'].max()

user_movie = np.zeros((users,movies))

for row in ratings.itertuples():
    user_movie[row.User-1, row.Movie-1] = row.Rate

km = KMeans(n_clusters=3, random_state=4)
km.fit(user_movie)
sep_group = km.predict(user_movie)

Groups = {i: user_movie[sep_group == i] for i in range(3)}


np.set_printoptions(threshold=np.inf)
for num, data in Groups.items():
    print(f"Groups {num} has {data.shape[0]} users.")
    

result = {
    'Avg': {},
    'AU': {},
    'SC': {},
    'AV': {},
    'BC': {},
    'CR': {}
}


for group_id, group in Groups.items():
    for alg in [Avg, AU, SC, AV, BC, CR]:
        scores = alg(group)
        result[alg.__name__][group_id] = top_sort(scores) + 1

for alg, groups in result.items():
    print(alg.__name__)
    for group_id, top_items in groups.items():
        print(f" Group {group_id}: Recommend: {top_items}")
