# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/12 11:42
@Auth ： Xiaoyao.Jia
@File ：shap.py
@IDE ：PyCharm
"""
# -*- coding: utf-8 -*-
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
from catboost import CatBoostClassifier
random_seed = 12345
splits_num = 5

DATA_PATH = '../data_/prodata.csv'
OUTPUT_DIR = '../result_shap'
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_pd = pd.read_csv(DATA_PATH, encoding='utf-8')
y = train_pd['mal']
X = train_pd.drop(columns=['mal', 'id'])
groups = train_pd['id']

max_features = X.shape[1]
base_model = CatBoostClassifier(
    random_state=random_seed,
    verbose=0
)
skf = StratifiedGroupKFold(
    n_splits=splits_num,
    shuffle=True,
    random_state=random_seed
)
fold_accuracy = []
fold_precision = []
fold_recall = []
fold_specificity = []
fold_f1 = []
fold_auc = []
fold_mcc = []
fold_records = []
all_shap_values = np.zeros((len(X), X.shape[1]), dtype=float)
all_base_values = np.zeros(len(X), dtype=float)
class Winsorizer:

    def __init__(self, lower=0.01, upper=0.01):
        self.lower = lower
        self.upper = upper

    def fit(self, X):
        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(1 - self.upper)
        return self

    def transform(self, X):
        return X.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1
        )
for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y, groups=groups)):
    print(f'\n========== Fold {k + 1} / {splits_num} ==========')
    X_train = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()
    y_train = y.iloc[train_index].copy()
    y_test = y.iloc[test_index].copy()
    winsorizer = Winsorizer(lower=0.01,upper=0.01)
    winsorizer.fit(X_train)
    X_train = winsorizer.transform(X_train)
    X_test = winsorizer.transform(X_test)
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train),columns=X.columns,index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test),columns=X.columns,index=X_test.index)
    model = clone(base_model)
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_value = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred,labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fold_accuracy.append(acc)
    fold_precision.append(pre)
    fold_recall.append(rec)
    fold_specificity.append(specificity)
    fold_f1.append(f1)
    fold_auc.append(auc_value)
    fold_mcc.append(mcc)
    print(f'Accuracy    : {acc:.6f}')
    print(f'Precision   : {pre:.6f}')
    print(f'Recall      : {rec:.6f}')
    print(f'Specificity : {specificity:.6f}')
    print(f'F1-score    : {f1:.6f}')
    print(f'AUC         : {auc_value:.6f}')
    print(f'MCC         : {mcc:.6f}')
    explainer = shap.TreeExplainer(model)
    shap_fold = explainer(X_test)
    fold_values = np.asarray(shap_fold.values)
    if fold_values.ndim != 2:
        raise ValueError(
            f'Unexpected SHAP values shape: {fold_values.shape}'
        )
    all_shap_values[test_index, :] = fold_values
    fold_base = np.asarray(shap_fold.base_values)
    if fold_base.ndim == 0:
        all_base_values[test_index] = float(fold_base)
    else:
        all_base_values[test_index] = fold_base.reshape(-1)
    fold_records.append({
        'fold': k + 1,
        'accuracy': acc,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'test_index': test_index,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
print('\n\n========== 5-fold Mean Performance ==========')
metrics_dict = {
    'Accuracy': fold_accuracy,
    'Precision': fold_precision,
    'Recall': fold_recall,
    'Specificity': fold_specificity,
    'F1-score': fold_f1,
    'AUC': fold_auc,
    'MCC': fold_mcc
}

summary_rows = []
for metric_name, values in metrics_dict.items():
    values = np.asarray(values, dtype=float)
    mean_value = np.nanmean(values)
    std_value = np.nanstd(values, ddof=1)
    print(
        f'{metric_name:<12}: '
        f'{mean_value:.6f} ± {std_value:.6f}'
    )
    summary_rows.append({
        'Metric': metric_name,
        'Mean': mean_value,
        'Std': std_value
    })
fold_results_df = pd.DataFrame({
    'Fold': np.arange(1, splits_num + 1),
    'Accuracy': fold_accuracy,
    'Precision': fold_precision,
    'Recall': fold_recall,
    'Specificity': fold_specificity,
    'F1-score': fold_f1,
    'AUC': fold_auc,
    'MCC': fold_mcc
})
fold_results_df.to_csv(
    os.path.join(OUTPUT_DIR, '5fold_performance_each_fold.csv'),
    index=False,
    encoding='utf-8-sig'
)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, '5fold_performance_mean_std.csv'),
    index=False,
    encoding='utf-8-sig'
)
shap_values_all = shap.Explanation(
    values=all_shap_values,
    base_values=all_base_values,
    data=X.values,
    feature_names=X.columns.tolist()
)
mean_abs_shap = np.abs(all_shap_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean_ABS_SHAP': mean_abs_shap
}).sort_values(
    by='Mean_ABS_SHAP',
    ascending=False
)

shap_importance_df.to_csv(
    os.path.join(OUTPUT_DIR, '5fold_mean_abs_SHAP.csv'),
    index=False,
    encoding='utf-8-sig'
)

print('\nTop 10 SHAP features:')
print(shap_importance_df.head(10))
from scipy.stats import gaussian_kde
top3_features = (
    shap_importance_df
    .sort_values('Mean_ABS_SHAP', ascending=False)
    ['Feature']
    .head(3)
    .tolist()
)
print('\nTop 3 SHAP features:')
for rank, feature in enumerate(top3_features, start=1):
    importance = shap_importance_df.loc[
        shap_importance_df['Feature'] == feature,
        'Mean_ABS_SHAP'
    ].iloc[0]

    print(
        f'{rank}. {feature}: '
        f'{importance:.6f}'
    )
fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
    dpi=300
)
interaction_feature = 'blood_flow'
for i, feature in enumerate(top3_features):

    # 当前特征在X中的列位置
    feature_index = X.columns.get_loc(feature)

    # 当前特征全部样本的原始取值
    feature_values = (
        X[feature]
        .astype(float)
        .to_numpy()
    )

    # 当前特征对应的5折 OOF SHAP 值
    feature_shap_values = all_shap_values[:, feature_index]
    ax1 = axes[0, i]
    benign_values = (
        X.loc[y == 0, feature]
        .dropna()
        .astype(float)
        .to_numpy()
    )
    malignant_values = (
        X.loc[y == 1, feature]
        .dropna()
        .astype(float)
        .to_numpy()
    )
    all_values = (
        X[feature]
        .dropna()
        .astype(float)
        .to_numpy()
    )
    value_min = np.min(all_values)
    value_max = np.max(all_values)
    value_range = value_max - value_min
    if value_range == 0:
        value_range = 1.0
    x_grid = np.linspace(
        value_min - 0.30 * value_range,
        value_max + 0.30 * value_range,
        400
    )
    if (
        len(benign_values) > 1
        and np.std(benign_values) > 0
    ):

        kde_benign = gaussian_kde(
            benign_values,
            bw_method=0.25
        )

        density_benign = kde_benign(x_grid)

        ax1.plot(
            x_grid,
            density_benign,
            color='salmon',
            linewidth=1.4
        )

        ax1.fill_between(
            x_grid,
            0,
            density_benign,
            color='salmon',
            alpha=0.45,
            label='benign'
        )

    else:
        ax1.axvline(
            benign_values[0],
            color='salmon',
            linewidth=2,
            label='benign'
        )
    if (
        len(malignant_values) > 1
        and np.std(malignant_values) > 0
    ):

        kde_malignant = gaussian_kde(
            malignant_values,
            bw_method=0.25
        )

        density_malignant = kde_malignant(x_grid)

        ax1.plot(
            x_grid,
            density_malignant,
            color='royalblue',
            linewidth=1.4
        )

        ax1.fill_between(
            x_grid,
            0,
            density_malignant,
            color='royalblue',
            alpha=0.40,
            label='malignant'
        )

    else:

        ax1.axvline(
            malignant_values[0],
            color='royalblue',
            linewidth=2,
            label='malignant'
        )


    ax1.set_xlabel(
        feature,
        fontsize=12
    )

    # 与你的示例图保持一致
    ax1.set_ylabel(
        'Density',
        fontsize=12
    )

    ax1.legend(
        loc='upper right',
        fontsize=10,
        frameon=True
    )

    ax1.tick_params(
        axis='both',
        labelsize=10
    )

    ax2 = axes[1, i]

    if interaction_feature in X.columns:

        shap.dependence_plot(
            feature,
            all_shap_values,
            X,
            interaction_index=interaction_feature,
            ax=ax2,
            show=False
        )

    else:

        shap.dependence_plot(
            feature,
            all_shap_values,
            X,
            interaction_index='auto',
            ax=ax2,
            show=False
        )


    ax2.set_xlabel(
        feature,
        fontsize=12
    )

    ax2.set_ylabel(
        'SHAP value for\n' + feature,
        fontsize=12
    )

    ax2.tick_params(
        axis='both',
        labelsize=10
    )


panel_labels = ['(a)', '(b)', '(c)']

for i, feature in enumerate(top3_features):

    axes[1, i].text(
        0.5,
        -0.24,
        f'{panel_labels[i]} {feature}',
        transform=axes[1, i].transAxes,
        ha='center',
        va='top',
        fontsize=15,
        fontfamily='Times New Roman'
    )

plt.subplots_adjust(
    left=0.06,
    right=0.96,
    top=0.97,
    bottom=0.12,
    wspace=0.35,
    hspace=0.28
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        'Top3_SHAP_KDE_dependence.png'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


plt.figure(figsize=(10, max(8, max_features * 0.45)))

shap.plots.bar(
    shap_values_all,
    max_display=max_features,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, 'SHAP_5fold_bar.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()

plt.figure(figsize=(10, max(8, max_features * 0.45)))

shap.plots.beeswarm(
    shap_values_all,
    max_display=max_features,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, 'SHAP_5fold_beeswarm.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()

clustering = shap.utils.hclust(X, y)

plt.figure(figsize=(10, max(8, max_features * 0.45)))

shap.plots.bar(
    shap_values_all,
    clustering=clustering,
    max_display=max_features,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, 'SHAP_5fold_clustering.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


mean_accuracy = np.mean(fold_accuracy)

representative_fold_index = int(
    np.argmin(
        np.abs(
            np.asarray(fold_accuracy) - mean_accuracy
        )
    )
)

rep = fold_records[representative_fold_index]

print('\n========== Representative Fold ==========')
print('Mean Accuracy          :', mean_accuracy)
print('Representative Fold    :', rep['fold'])
print('Representative Accuracy:', rep['accuracy'])



rep_model = rep['model']
rep_X = rep['X_test']
rep_y = rep['y_test']

rep_explainer = shap.TreeExplainer(rep_model)
rep_shap = rep_explainer(rep_X)


positive_positions = np.where(
    rep_y.to_numpy() == 1
)[0]

if len(positive_positions) > 0:

    positive_pos = positive_positions[0]

    shap.plots.waterfall(
        rep_shap[positive_pos],
        max_display=max_features,
        show=False
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            'SHAP_representative_malignant_waterfall.png'
        ),
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()


negative_positions = np.where(
    (rep_y.to_numpy() == 0))[0]

if len(negative_positions) > 0:

    negative_pos = negative_positions[0]

    shap.plots.waterfall(
        rep_shap[negative_pos],
        max_display=max_features,
        show=False
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            'SHAP_representative_benign_waterfall.png'
        ),
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()


rep_test_index = rep['test_index']


rep_groups = (
    groups
    .iloc[rep_test_index]
    .reset_index(drop=True)
)


decision_base_value = float(
    np.asarray(rep_shap.base_values)
    .reshape(-1)[0]
)


decision_feature_range = slice(None, None, -1)


decision_cmap = plt.get_cmap('coolwarm')



rng = np.random.default_rng(random_seed)

unique_patient_ids = rep_groups.drop_duplicates().to_numpy()

n_random_patients = min(
    20,
    len(unique_patient_ids)
)

selected_patient_ids = rng.choice(
    unique_patient_ids,
    size=n_random_patients,
    replace=False
)

random_positions = []

for patient_id in selected_patient_ids:


    patient_positions = np.where(
        rep_groups.to_numpy() == patient_id
    )[0]


    selected_position = int(
        rng.choice(patient_positions)
    )

    random_positions.append(selected_position)

random_positions = np.asarray(
    random_positions,
    dtype=int
)

print(
    f'Random decision plot patients: '
    f'{len(random_positions)}'
)



random_decision_path = os.path.join(
    OUTPUT_DIR,
    'SHAP_decision_random20_patients.png'
)

plt.figure(
    figsize=(7, 10),
    dpi=300
)

shap.decision_plot(
    decision_base_value,
    rep_shap.values[random_positions, :],
    features=rep_X.iloc[random_positions, :],
    feature_names=X.columns.tolist(),
    feature_order='importance',
    feature_display_range=decision_feature_range,
    link='identity',
    plot_color=decision_cmap,
    color_bar=True,
    auto_size_plot=False,
    title='Random 20 patients',
    show=False
)

plt.tight_layout()

plt.savefig(
    random_decision_path,
    dpi=400,
    bbox_inches='tight'
)

plt.close()



malignant_decision_path = os.path.join(
    OUTPUT_DIR,
    'SHAP_decision_malignant_case.png'
)

if len(positive_positions) > 0:

    # 与前面的 malignant waterfall 使用同一个病例
    malignant_position = int(positive_pos)

    print(
        'Malignant decision case: '
        f'row_position={malignant_position}, '
        f'original_index={rep_X.index[malignant_position]}, '
        f'patient_id={rep_groups.iloc[malignant_position]}'
    )

    plt.figure(
        figsize=(7, 10),
        dpi=300
    )

    shap.decision_plot(
        decision_base_value,
        rep_shap.values[malignant_position, :],
        features=rep_X.iloc[malignant_position, :],
        feature_names=X.columns.tolist(),
        feature_order='importance',
        feature_display_range=decision_feature_range,
        link='identity',
        plot_color=decision_cmap,
        color_bar=True,
        auto_size_plot=False,
        title='Malignant case',
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        malignant_decision_path,
        dpi=400,
        bbox_inches='tight'
    )

    plt.close()


benign_decision_path = os.path.join(
    OUTPUT_DIR,
    'SHAP_decision_benign_case.png'
)

if len(negative_positions) > 0:

    benign_position = int(negative_pos)

    print(
        'Benign decision case: '
        f'row_position={benign_position}, '
        f'original_index={rep_X.index[benign_position]}, '
        f'patient_id={rep_groups.iloc[benign_position]}'
    )

    plt.figure(
        figsize=(7, 10),
        dpi=300
    )

    shap.decision_plot(
        decision_base_value,
        rep_shap.values[benign_position, :],
        features=rep_X.iloc[benign_position, :],
        feature_names=X.columns.tolist(),
        feature_order='importance',
        feature_display_range=decision_feature_range,
        link='identity',
        plot_color=decision_cmap,
        color_bar=True,
        auto_size_plot=False,
        title='Benign case',
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        benign_decision_path,
        dpi=400,
        bbox_inches='tight'
    )

    plt.close()


decision_images = []
decision_titles = []

if os.path.exists(random_decision_path):
    decision_images.append(
        plt.imread(random_decision_path)
    )
    decision_titles.append(
        '(a) Random 20 patients'
    )

if os.path.exists(malignant_decision_path):
    decision_images.append(
        plt.imread(malignant_decision_path)
    )
    decision_titles.append(
        '(b) Malignant case'
    )

if os.path.exists(benign_decision_path):
    decision_images.append(
        plt.imread(benign_decision_path)
    )
    decision_titles.append(
        '(c) Benign case'
    )

if len(decision_images) > 0:

    fig, axes = plt.subplots(
        1,
        len(decision_images),
        figsize=(7 * len(decision_images), 10),
        dpi=300
    )

    if len(decision_images) == 1:
        axes = [axes]

    for ax, image, title in zip(
        axes,
        decision_images,
        decision_titles
    ):

        ax.imshow(image)
        ax.axis('off')

        ax.set_title(
            title,
            fontsize=15,
            fontfamily='Times New Roman',
            pad=12
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            'SHAP_decision_paths_combined.png'
        ),
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()



print('\\n全部完成。')
print('结果保存目录：', OUTPUT_DIR)
