# DecisionTreeClassifier

决策树分类器。

```ts
interface DecisionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    criterion?: 'entropy' | 'gini';
}

public constructor(props: DecisionTreeProps = {})
```

## 参数

+ `max_depth`, 最大深度，指决策树的最大深度，建树时会判断大于该数值时停止对当前节点继续划分。默认值为`Infinity`。
+ `min_samples_split`, 最小样本分割阈值，当叶子节点包含的样本数小于该值时，停止对节点的继续划分。
+ `criterion`, 不纯度的计算方式，可选值为`'entropy'`和`'gini'`，默认为entropy。

## APIs

### fit(sampleX: number[][], sampleY: number[]): void
训练模型，根据给出的数据建立一颗决策树。目前暂时不支持持续构建，默认当此方法被调用时，会重新建立一颗新的决策树。

### predict(testX: number[][]): number[]
对测试样本进行预测，给出预测的分类标签。

## 用例

```ts
const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
const y = [-1, -1, -1, 1, 1, 1]
const T = [[-1, -1], [2, 2], [3, 2]]

const dtree = new DecisionTreeClassifier({ criterion: 'gini' });
dtree.fit(X, y);
const ans = dtree.predict(T);
// ans = [-1, 1, 1]
console.log(ans)
```

## Reference
+ [https://scikit-learn.org/stable/modules/tree.html#tree](https://scikit-learn.org/stable/modules/tree.html#tree)