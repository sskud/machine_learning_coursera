import pandas
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from time import time
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import arange
from sklearn.linear_model import LogisticRegression

features = pandas.read_csv('features.csv')
y = features.radiant_win.values

features.drop([
    'duration',
    'radiant_win',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire'
], axis=1, inplace=True)

#Признаки с пропусками
nnull = features.count()
length = len(features)
dset = nnull[nnull<length]
print dset

features.fillna(0, inplace=True)
X = features.ix[:, :]

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=1) #генерация разбиений

#Градиентный бустинг
for trees in [10, 20, 30, 40, 50]:
    t = time()
    clf = GradientBoostingClassifier(n_estimators=trees)
    clf.fit(X, y)
    score = cross_val_score(clf, X, y, cv=kf, scoring='roc_auc').mean()
    t = time() - t
    print 'Trees: %i, score: %.4f, time: %.2f s' % (trees, score, t)

scaler = StandardScaler().fit(features)
X = scaler.transform(X)

#Логистическая регрессия
def log_reg(X):
    C_arr = [10 ** x for x in arange(-3, 3, 1)]
    best_score, best_c, best_time = 0, 0, 0   
    for C in C_arr:
        t = time()
        score = cross_val_score(LogisticRegression(C=C), X, y, scoring='roc_auc', cv=kf).mean()
        t = time() - t
        print 'Score: %.4f, C: %.4f, time: %.2f s' % (score, C, t)
        if score > best_score:
            best_score = score
            best_c = C
            best_time = t  
    print 'Best score: %.4f, C: %.4f, time: %.2f s' % (best_score, best_c, best_time)

#Вызов логистической регрессии с над всеми исходными признаками
log_reg(X)

new_features = pandas.read_csv('features.csv', index_col='match_id')

new_features.drop([
    'duration',
    'radiant_win',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire',
    'lobby_type',
    'r1_hero',
    'r2_hero',
    'r3_hero',
    'r4_hero',
    'r5_hero',
    'd1_hero',
    'd2_hero',
    'd3_hero',
    'd4_hero',
    'd5_hero'
], axis=1, inplace=True)

new_features.fillna(0, inplace=True)
new_X = new_features.ix[:, :]
scaler = StandardScaler().fit(new_features)
new_X = scaler.transform(new_X)

#Вызов логистической регрессии, без учета категориальных признаков
log_reg(new_X)

heroes = pandas.Series()

for h in ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']:
    heroes = heroes.append(features[h])

print '\nUnique heroes: %i' % len(heroes.unique())

N = heroes.max()
X_pick = np.zeros((features.shape[0], N))

for i, match_id in enumerate(features.index):
    for p in xrange(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

X = np.hstack([scaler.transform(new_X), X_pick])

#Вызов логистической регрессии с учетом «мешка слов» 
log_reg(X)

features_test = pandas.read_csv('features_test.csv', index_col='match_id')

features_test.fillna(0, inplace=True)

X_pick_test = np.zeros((features_test.shape[0], N))

for i, match_id in enumerate(features_test.index):
    for p in xrange(5):
        X_pick_test[i, features_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

features_test.drop([
    'lobby_type',
    'r1_hero',
    'r2_hero',
    'r3_hero',
    'r4_hero',
    'r5_hero',
    'd1_hero',
    'd2_hero',
    'd3_hero',
    'd4_hero',
    'd5_hero'
], axis=1, inplace=True)

scaler = StandardScaler().fit(new_features)

X_test = features_test.ix[:, :]

logisticRegression = LogisticRegression(C=100)
logisticRegression.fit(np.hstack([scaler.transform(new_features), X_pick]), y)

scaler = StandardScaler().fit(features_test)

results = pandas.DataFrame(
    index=features_test.index,
    data=logisticRegression.predict_proba(np.hstack((scaler.transform(features_test), X_pick_test)))[:, 1],
    columns=['radiant_win']
)

results.to_csv('predictions.csv')

print '\nMin and max predicts: %.4f %.4f' % (np.min(results), np.max(results))
    