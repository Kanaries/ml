from sklearn import tree
import random
sample_size = 100
trainX = list(map(lambda i: [random.randint(0, 10), random.randint(0, 10), random.randint(0, 5)] ,range(20)))
# X = [[0, 0], [2, 2], [3, 6]]
# y = [0.5, 2.5, 3.6]
trainy = list(map(lambda x: x[0] * 10 + x[1] * 20 - x[2] * 12, trainX))

testX = list(map(lambda i: [random.randint(0, 10), random.randint(0, 10), random.randint(0, 5)] ,range(10)))
# X = [[0, 0], [2, 2], [3, 6]]
# y = [0.5, 2.5, 3.6]
# testy = list(map(lambda x: x[1] * 10 + x[2] * 20 - x[3] * 12, testX))
clf = tree.DecisionTreeRegressor(splitter="best")
clf = clf.fit(trainX, trainy)
result = clf.predict(testX)

print(result)
print('==trainX=='*10)
print(trainX)
print('==trainy==='*10)
print(trainy)
print('===testX==='*10)
print(testX)


