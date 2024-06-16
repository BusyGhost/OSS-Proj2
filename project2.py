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
    return np.sum(np.where(group > 0, 1, 0), axis=0)


def AV(group, threshold=4):
    return np.sum(np.where(group >= threshold, group, 0), axis=0)


def BC(group):
    data = pd.DataFrame(group)
    data = data.rank(axis=1) - 1
    data = data.sum(axis=0)
    return data.values


def CR(group):
    user_num = group.shape[0]
    movie_num = group.shape[1]
    result = np.zeros(movie_num)
    for i in range(movie_num):
        for j in range(i + 1, movie_num):
            beat_num = (group[:, i] > group[:, j]).sum()
            lose_num = (group[:, i] < group[:, j]).sum()
            temp = np.sign(beat_num - lose_num)
            result[i] += temp
            result[j] -= temp
    return result


# 각 그룹에 대해 알고리즘 실행 및 결과 반환
def compute(groups):
    result = np.zeros((3, 6), dtype=object)

    for group_id, group in groups.items():
        for idx, alg in enumerate([Avg, AU, SC, AV, BC, CR]):
            scores = alg(group)
            movies = top_sort(scores) + 1
            result[group_id, idx] = movies
            print(f"\nGroup{group_id}, {alg.__name__} computation complete\n Recommend: {movies}")
            
    return result

# 데이터 로드 및 처리
ratings = pd.read_csv('ratings.dat', sep='::', engine='python', names=['User', 'Movie', 'Rate', 'time'])

users = ratings['User'].max()
movies = ratings['Movie'].max()

user_movie = np.zeros((users, movies))

for row in ratings.itertuples():
    user_movie[row.User - 1, row.Movie - 1] = row.Rate

km = KMeans(n_clusters=3, random_state=4, n_init='auto')
sep_group = km.fit_predict(user_movie)

# 그룹화
Groups = {i+1 : user_movie[sep_group == i] for i in range(3)}

# 그룹 정보 출력
np.set_printoptions(threshold=np.inf)
for num, data in Groups.items():
    print(f"Group {num} has {data.shape[0]} users.")

# 계산 실행
table = compute(Groups)

# 결과 출력
algorithms = ['Avg', 'AU', 'SC', 'AV', 'BC', 'CR']
for group_id in range(1,4):
    print(f"\nGroup {group_id}")
    for alg_id, alg in enumerate(algorithms):
        print(f"{alg}: {table[group_id, alg_id]}")
