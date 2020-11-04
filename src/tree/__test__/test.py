from sklearn import tree
X = [[0, 0], [2, 2], [3, 6]]
y = [0.5, 2.5, 3.6]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
result = clf.predict([[1, 1]])

print(result)
