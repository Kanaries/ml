## DecisionTreeRegressor

```ts
interface RegressionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
}

constructor(props: RegressionTreeProps = {})
```

### Example
```ts
const reg = new DecisionTreeRegressor({ max_depth: 5 });
reg.fit(X, Y);
const preds = reg.predict(T);
```
