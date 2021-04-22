from sklearn import tree
import random
import graphviz

print(graphviz)
sample_size = 100
# trainX = list(map(lambda i: [random.randint(0, 10), random.randint(0, 10), random.randint(0, 5)] ,range(20)))
trainX = [
    [6, 4, 0],
    [4, 0, 5],
    [6, 2, 0],
    [8, 3, 0],
    [9, 2, 2],
    [1, 1, 5],
    [2, 2, 4],
    [9, 10, 1],
    [7, 6, 3],
    [5, 6, 4],
    [8, 2, 4],
    [6, 0, 5],
    [1, 9, 1],
    [3, 0, 4],
    [4, 5, 0],
    [4, 7, 2],
    [3, 1, 3],
    [9, 8, 0],
    [2, 7, 0],
    [4, 8, 4],
]
# X = [[0, 0], [2, 2], [3, 6]]
# y = [0.5, 2.5, 3.6]
# trainy = list(map(lambda x: x[0] * 10 + x[1] * 20 - x[2] * 12, trainX))
trainy = [140, -20, 100, 140, 106, -30, 12, 278, 154,
          122, 72, 0, 178, -18, 140, 156, 14, 250, 160, 152]

# testX = list(map(lambda i: [random.randint(0, 10), random.randint(0, 10), random.randint(0, 5)] ,range(10)))
testX = [
        [9, 4, 1],
        [3, 0, 4],
        [5, 5, 2],
        [7, 2, 5],
        [7, 3, 3],
        [4, 2, 5],
        [3, 3, 1],
        [2, 4, 2],
        [1, 7, 3],
        [6, 8, 2],
]
# X = [[0, 0], [2, 2], [3, 6]]
# y = [0.5, 2.5, 3.6]
# testy = list(map(lambda x: x[1] * 10 + x[2] * 20 - x[3] * 12, testX))
clf = tree.DecisionTreeRegressor(splitter="best")
clf = clf.fit(trainX, trainy)
result = clf.predict(testX)

tree.plot_tree(clf)
print(result)
# print('==trainX=='*10)
# print(trainX)
# print('==trainy==='*10)
# print(trainy)
# print('===testX==='*10)
# print(testX)


