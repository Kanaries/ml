## Algebra

### Algebra.gaussianElimination(高斯消元)
```ts
function gaussianElimination(A: number[][]): number[][] | false
```
参数：
+ `A`: 原始的矩阵

返回：返回一个行阶梯型的矩阵，且主元为1。

```ts
const A = [
    [0, -1, -1, 1],
    [1, 1, 1, 1],
    [2, 4, 1, -2],
    [3, 1, -2, 2]
]

const result = gaussianElimination(A);
// result = [
//     [1, 1, 1, 1],
//     [0, 1, 1, -1],
//     [0, 0, 1, 2 / 3],
//     [0, 0, 0, 1],
// ];
```

### Algebra.identityMatrix (单位矩阵)
```ts
function identityMatrix(A: number[][]): number[][]
```

返回一个对角线上值为1的方形矩阵

### Algebra.Inverse (矩阵求逆)
矩阵求逆

#### elementary 基础方法
使用高斯消元法求解逆矩阵
```ts
elementary (A: number[][]): number[][] | false
```
+ 参数：$ A $
+ 返回：$ A^{-1} $ 或`false`


### Algebra.transpose (矩阵转置)
```ts
transpose(matrix: number[][]): number[][]
```

### Algebra.augmentMatrix (增广矩阵)
```ts
augmentMatrix(matrix: number[][], expanded: number[][]): number[][]
```