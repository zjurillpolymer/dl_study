"""
Random Forest 入门示例
======================
用 Iris 数据集做分类 + 回归对比 + 特征重要性可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# ============================================================
# 1. 分类任务 —— Iris 数据集
# ============================================================
print("=" * 50)
print("1. 分类任务 —— Iris 数据集")
print("=" * 50)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=100,      # 100 棵树
    max_depth=3,           # 限制树深，防止过拟合
    min_samples_leaf=4,    # 叶节点最少样本数
    random_state=42,
    oob_score=True,        # 用袋外样本评估
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.2%}")
print(f"OOB 评分:     {clf.oob_score_:.2%}")
print()

# ============================================================
# 2. 特征重要性
# ============================================================
print("特征重要性 (Gini impurity reduction):")
for name, imp in zip(feature_names, clf.feature_importances_):
    print(f"  {name:20s}  {imp:.3f}")

# ============================================================
# 3. 回归任务 —— 人造数据
# ============================================================
print()
print("=" * 50)
print("2. 回归任务")
print("=" * 50)

X_r, y_r = make_regression(
    n_samples=500, n_features=5, noise=15, random_state=42
)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_r, y_r, test_size=0.3, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42,
)
reg.fit(Xr_train, yr_train)

yr_pred = reg.predict(Xr_test)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
print(f"回归 RMSE: {rmse:.2f}")

# ============================================================
# 4. 可视化 —— 单棵决策树 vs 随机森林的决策边界
# ============================================================
# 只用前两个特征画图方便展示
X_2d = X[:, :2]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d, y, test_size=0.3, random_state=42
)

# 单棵决策树
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X2_train, y2_train)

# 随机森林
rf_2d = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
rf_2d.fit(X2_train, y2_train)

# 画决策边界
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

for ax, model, title in zip(axes, [tree, rf_2d],
                              ["单棵决策树 (depth=3)", "随机森林 (50棵树, depth=3)"]):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set2")
    for cls in range(3):
        idx = y2_test == cls
        ax.scatter(X2_test[idx, 0], X2_test[idx, 1],
                   c=[["red", "green", "blue"][cls]],
                   label=class_names[cls], edgecolors="k", s=30)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("random_forest_demo.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print("决策边界图已保存 → random_forest_demo.png")
print()

# ============================================================
# 5. n_estimators 对性能的影响
# ============================================================
print("=" * 50)
print("3. 树的数量对准确率的影响")
print("=" * 50)

n_trees_range = [1, 5, 10, 20, 50, 100, 200, 500]
scores = []
for n in n_trees_range:
    rf = RandomForestClassifier(
        n_estimators=n, max_depth=3, random_state=42
    )
    rf.fit(X_train, y_train)
    s = accuracy_score(y_test, rf.predict(X_test))
    scores.append(s)
    print(f"  n_estimators={n:3d}  →  acc={s:.2%}")

# 画曲线
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(n_trees_range, scores, "o-", color="steelblue")
ax.set_xlabel("Number of trees (n_estimators)")
ax.set_ylabel("Test accuracy")
ax.set_xscale("log")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("random_forest_trees_vs_acc.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print("收敛曲线已保存 → random_forest_trees_vs_acc.png")
