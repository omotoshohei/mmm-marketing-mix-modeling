import numpy as np
import pandas as pd
import tqdm
import joblib

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split

from scipy.optimize import basinhopping
from scipy.optimize import Bounds

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.logging.disable_default_handler()

from functools import partial
from datetime import timedelta

import matplotlib.pyplot as plt
plt.style.use('ggplot') #グラフスタイル
plt.rcParams['figure.figsize'] = [10, 5] #グラフサイズ
plt.rcParams['font.size'] = 10 #フォントサイズ
import japanize_matplotlib #グラフ内で日本語利用

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
warnings.simplefilter('ignore')

##########################
# 時系列特徴量
##########################

#
# 三角関数特徴量の生成関数を定義
#

def add_fourier_terms(df, num, seasonal):
    '''
    引数df: 元のデータフレーム
    引数num: フーリエ項の数（基底の数）、sinとcosのセット数
    引数seasonal: フーリエ変換の周期
    戻り値: フーリエ項を追加後のデータフレーム
    '''
    # t列がdfにない場合、t列を0からdfの長さまでの連番で作成
    if 't' not in df.columns:
        df['t'] = pd.RangeIndex(start=0, stop=len(df))

    # 三角関数特徴量(フーリエ項)を追加
    for i in range(1, num + 1):
        # sin項を追加
        df['sin_' + str(i)] = np.sin(i * 2 * np.pi * df.t / seasonal)
        # cos項を追加
        df['cos_' + str(i)] = np.cos(i * 2 * np.pi * df.t / seasonal)

    return df

#
# 三角関数の列をそのまま利用して特徴量との交互作用項を生成
#

def create_seasonal_interaction_terms(df, X_col, features):
    # pd.DataFrameへのキャストを追加
    df = pd.DataFrame(df, columns=X_col)
    # 三角関数の項と指定された特徴量との交互作用項を生成
    for feature in features:
        if feature in df.columns:  # 列が存在するか確認
            for trig_func in df.filter(regex='^sin_|^cos_').columns:
                df[f'{feature}_{trig_func}'] = df[feature] * df[trig_func]
    return df
    
##########################
# アドストック
##########################

#
# キャリオーバー効果
#

# キャリオーバー効果関数
def carryover_advanced(X: np.ndarray, length, peak, rate1, rate2, c1, c2):
    X = np.append(np.zeros(length-1), X)
    
    Ws = np.zeros(length)
    
    for l in range(length):
        if l<peak-1:
            W = rate1**(abs(l-peak+1)**c1)
        else:
            W = rate2**(abs(l-peak+1)**c2)
        Ws[length-1-l] = W
    
    carryover_X = []
    
    for i in range(length-1, len(X)):
        X_array = X[i-length+1:i+1]
        Xi = sum(X_array * Ws)/sum(Ws)
        carryover_X.append(Xi)
        
    return np.array(carryover_X)

# キャリオーバー効果を適用するカスタム変換器クラス
class CustomCarryOverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, carryover_params=None):
        self.carryover_params = carryover_params if carryover_params is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        transformed_X = np.copy(X)
        for i, params in enumerate(self.carryover_params):
            transformed_X[:, i] = carryover_advanced(X[:, i], **params)
        return transformed_X

#
# 飽和効果
#

# 飽和関数（指数型）
def exponential_function(x, d):
    result = 1 - np.exp(-d * x)
    return result
    
# 飽和関数（ロジスティック曲線）
def logistic_function(x, L, k, x0):
    result = L / (1+ np.exp(-k*(x-x0)))  
    return result

# 飽和関数を適用するカスタム変換器クラス
class CustomSaturationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, curve_params=None):
        self.curve_params = curve_params if curve_params is not None else []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        transformed_X = np.copy(X)
        for i, params in enumerate(self.curve_params):
            saturation_function = params.pop('saturation_function') 
            if saturation_function == 'logistic':
                transformed_X[:, i] = logistic_function(X[:, i], **params)
            elif saturation_function == 'exponential':
                transformed_X[:, i] = exponential_function(X[:, i], **params)
            params['saturation_function'] = saturation_function # Returning the saturation_function back to params
        return transformed_X

#
# アドストックをグラフ表示
#

def plot_carryover_effect(params, feature_name, fig, axes, i):
    max_length = max(10, params['length'])
    x = np.concatenate(([1], np.zeros(max_length - 1)))
    y = carryover_advanced(x, **params)
    y = y / max(y)
    params_r = {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}
    axes[2*i].bar(np.arange(1, max_length + 1), y)
    axes[2*i].set_title(f'Carryover Effect for {feature_name}')
    axes[2*i].text(0, 1.1, params_r, ha='left',va='top')
    axes[2*i].set_xlabel('Carryover status (1 is implementation timing)')
    axes[2*i].set_ylabel('Effect')
    axes[2*i].set_xticks(range(len(y)))
    axes[2*i].set_ylim(0, 1.1)

def plot_saturation_curve(params, feature_name, fig, axes, i):
    x = np.linspace(-1, 3, 400)
    saturation_function = params.pop('saturation_function') 
    if saturation_function == 'logistic':
        y = logistic_function(x, **params)
    elif saturation_function == 'exponential':
        y = exponential_function(x, **params)
    params_r = {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}
    axes[2*i+1].plot(x, y, label=feature_name)
    axes[2*i+1].set_title(f'Saturation Curve for {feature_name}')
    params['saturation_function'] = saturation_function 
    axes[2*i+1].text(-1, max(y)* 1.1, params_r, ha='left',va='top')
    axes[2*i+1].set_xlabel('X')
    axes[2*i+1].set_ylabel('Transformation')
    axes[2*i+1].set_ylim(0, max(y) * 1.1)
    axes[2*i+1].set_xlim(-1, 3)

def plot_effects(carryover_params, curve_params, feature_names):
    fig, axes = plt.subplots(len(feature_names) * 2, 1, figsize=(12, 10*len(feature_names)))
    for i, params in enumerate(carryover_params):
        plot_carryover_effect(params, feature_names[i], fig, axes, i)
    for i, params in enumerate(curve_params):
        plot_saturation_curve(params, feature_names[i], fig, axes, i)
    plt.tight_layout()
    plt.show()

##########################
# MMM
##########################

#
# RegARIMAモデルのクラスの定義
#

class RegARIMAModel:
    def __init__(self, alpha=1.0):
        """
        RegARIMAモデルクラスを初期化
        :param alpha: Ridge回帰の正則化パラメータ
        """
        self.alpha = alpha
        self.ridge_model = Ridge(alpha=alpha)
        self.arima_model = None
        self.residuals = None

    def fit(self, X, y):
        """
        モデルをデータにフィットさせる
        :param X: 説明変数のndarray
        :param y: 目的変数のndarray
        """
        # Ridge回帰でフィット
        self.ridge_model.fit(X, y)
        
        # Ridge回帰の残差を計算
        predictions = self.ridge_model.predict(X)
        self.residuals = y - predictions
        
        # 残差に対してauto_arimaを実行
        self.arima_model = auto_arima(
            self.residuals, seasonal=False, stepwise=False,
            suppress_warnings=True, error_action="ignore", trace=False)

    def predict(self, X, n_periods=0):
        """
        モデルを使って予測する
        :param X: 説明変数のndarray
        :param n_periods: 予測する期間の数
        :return: 予測された値
        """
        # Ridge回帰からの予測
        ridge_predictions = self.ridge_model.predict(X)
        
        # ARIMAモデルからの未来予測
        if n_periods > 0:
            arima_predictions = self.arima_model.predict(n_periods=n_periods)
        else:
            arima_predictions = self.arima_model.predict_in_sample()
        
        # Ridgeの予測とARIMAの予測の和を返す
        # ここでは、n_periodsが1の場合を想定
        return ridge_predictions + arima_predictions[0]
 
#
# Optunaの目的関数
#

def regarima_objective(trial, X, y, apply_effects_features):

    carryover_params = []
    curve_params = []

    # 列名リストからインデックスのリストを作成
    apply_effects_indices = [X.columns.get_loc(column) for column in apply_effects_features]
    no_effects_indices = list(set(range(X.shape[1])) - set(apply_effects_indices))
    
    for feature in apply_effects_features:
        carryover_length = trial.suggest_int(f'carryover_length_{feature}', 1, 10)
        carryover_peak = trial.suggest_int(f'carryover_peak_{feature}', 1, carryover_length)
        carryover_rate1 = trial.suggest_float(f'carryover_rate1_{feature}', 0, 1)
        carryover_rate2 = trial.suggest_float(f'carryover_rate2_{feature}', 0, 1)
        carryover_c1 = trial.suggest_float(f'carryover_c1_{feature}', 0, 2)
        carryover_c2 = trial.suggest_float(f'carryover_c2_{feature}', 0, 2)
        carryover_params.append({
            'length': carryover_length, 
            'peak': carryover_peak, 
            'rate1': carryover_rate1, 
            'rate2': carryover_rate2, 
            'c1': carryover_c1, 
            'c2': carryover_c2,
        })

        saturation_function = trial.suggest_categorical(f'saturation_function_{feature}', ['logistic', 'exponential'])
        if saturation_function == 'logistic':
            curve_param_L = trial.suggest_float(f'curve_param_L_{feature}', 0, 10)
            curve_param_k = trial.suggest_float(f'curve_param_k_{feature}', 0, 10)
            curve_param_x0 = trial.suggest_float(f'curve_param_x0_{feature}', 0, 2)
            curve_params.append({
                'saturation_function': saturation_function,
                'L': curve_param_L,
                'k': curve_param_k,
                'x0': curve_param_x0,
            })
        elif saturation_function == 'exponential':
            curve_param_d = trial.suggest_float(f'curve_param_d_{feature}', 0, 10)
            curve_params.append({
                'saturation_function': saturation_function,
                'd': curve_param_d,
            })

    alpha = trial.suggest_float('alpha', 1e-3, 1e+3)

    preprocessor = ColumnTransformer(
        transformers=[
            ('effects', Pipeline([
                ('carryover', CustomCarryOverTransformer(carryover_params=carryover_params)),
                ('saturation', CustomSaturationTransformer(curve_params=curve_params))
            ]), apply_effects_indices),
            ('no_effects', 'passthrough', no_effects_indices)
        ],
        remainder='drop'
    )

    create_seasonal_interaction_terms_transformer = FunctionTransformer(
        create_seasonal_interaction_terms, 
        kw_args={
            'X_col': X.columns,
            'features': apply_effects_features
        }, 
        validate=False)

    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('preprocessor', preprocessor),
        ('create_seasonal', create_seasonal_interaction_terms_transformer),
        ('estimator', RegARIMAModel(alpha=alpha))
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse = np.mean([np.sqrt(-score) for score in scores])

    return rmse
    
#
# Optunaによる最適なハイパーパラメータの探索実行
#

def run_optimization(objective, X, y, apply_effects_features, n_trials=1000, study=None):

    # Optunaのスタディの作成と最適化の実行
    if study is None:
        study = optuna.create_study(direction='minimize')

    objective_with_data = partial(
        objective, 
        X=X, y=y, 
        apply_effects_features=apply_effects_features)
        
    study.optimize(
        objective_with_data, 
        n_trials=n_trials, 
        show_progress_bar=True)

    # 最適化の実行結果の表示
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    return study

#
# RegARIMAモデルの学習と予測
#

def train_and_evaluate_regarima(model, X, y, Xmax=1, ymax=1):
    """
    モデルを学習し、予測と評価を行う関数
    引数:
    - model: 学習するモデルのインスタンス
    - X: 特徴量のデータフレーム
    - y: ターゲット変数のデータフレーム
    戻り値:
    - 学習済みモデルmodel
    - 予測値pred（学習データ期間）
    """

    X_scaled = X/Xmax
    y_scaled = y/ymax

    # モデルの学習および予測（学習データ期間）
    model.fit(X_scaled, y_scaled)
    pred = model.predict(X_scaled)
    pred = pd.DataFrame(pred*ymax, index=X.index, columns=['y'])

    # 精度指標の計算
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    mape = mean_absolute_percentage_error(y, pred)
    r2 = r2_score(y, pred)

    # 実測値と予測値の時系列推移
    plot_actual_vs_predicted(y.index, y.values, pred.y.values, (0, np.max(y) * 1.1))

    # 精度指標の出力
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('MAPE:', mape)
    print('R2:', r2)

    return model,pred
    
#
# MMMパイプライン
#

def build_MMM_pipeline_regarima(X, y, apply_effects_features, carryover_params, curve_params, alpha=1):
    """
    MMMパイプラインを構築する関数
    
    引数:
    - X: DataFrame, 特徴量
    - y: Series, 目的変数
    - apply_effects_features: list, アドストックを考慮する特徴量
    - carryover_params: list, キャリーオーバー効果のパラメータ
    - curve_params: list, 飽和関数のパラメータ
    
    戻り値:
    - MMM_pipeline: Pipeline, MMMパイプライン
    - trained_model: Ridge, 学習済みモデル
    - pred: DataFrame, 学習済みモデルで予測した値
    """
    
    # 列名リストからインデックスのリストを作成
    apply_effects_indices = [X.columns.get_loc(column) for column in apply_effects_features]
    no_effects_indices = list(set(range(X.shape[1])) - set(apply_effects_indices))

    # ColumnTransformerを使って特定の列にのみ変換を適用する
    preprocessor = ColumnTransformer(
        transformers=[
            ('effects', Pipeline([
                ('carryover_transformer', CustomCarryOverTransformer(carryover_params=carryover_params)),
                ('saturation_transformer', CustomSaturationTransformer(curve_params=curve_params))
            ]), apply_effects_indices),
            ('no_effects', 'passthrough', no_effects_indices)
        ],
        remainder='drop'  # 効果適用外の特徴量は処理しない
    )

    # FunctionTransformerの作成
    create_seasonal_interaction_terms_transformer = FunctionTransformer(
        create_seasonal_interaction_terms, 
        kw_args={
            'X_col': X.columns,
            'features': apply_effects_features
        }, 
        validate=False)

    MMM_pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('preprocessor', preprocessor),
        ('create_seasonal', create_seasonal_interaction_terms_transformer),
        ('estimator', RegARIMAModel(alpha=alpha))
    ])

    return MMM_pipeline
    
#
# Optunaの結果（ハイパーパラメータ）からMMMパイプライン構築
#
    
def create_model_from_trial_regarima(trial, X, y, apply_effects_features):

    #
    # Optunaの実行結果からハイパーパラメータを取得
    #

    carryover_keys = ['length', 'peak', 'rate1', 'rate2', 'c1', 'c2']
    curve_keys_logistic = ['L', 'k', 'x0']
    curve_keys_exponential = ['d']

    # ハイパーパラメータを抽出しキーと値の辞書を作成
    def fetch_params(prefix, feature_name, params, trial_params):
        return {key: trial_params[f'{prefix}_{key}_{feature_name}'] for key in params}

    # キャリーオーバー効果関数ハイパーパラメータを抽出
    carryover_params = [fetch_params('carryover', feature_name, carryover_keys, trial.params) for feature_name in apply_effects_features]

    # 飽和関数ハイパーパラメータを抽出
    curve_params = []
    for feature_name in apply_effects_features:
        saturation_function = trial.params[f'saturation_function_{feature_name}']
        curve_param = {'saturation_function': saturation_function}
        
        if saturation_function == 'logistic':
            curve_param.update(fetch_params('curve_param', feature_name, curve_keys_logistic, trial.params))
        elif saturation_function == 'exponential':
            curve_param.update(fetch_params('curve_param', feature_name, curve_keys_exponential, trial.params))
        
        curve_params.append(curve_param)

    # 推定器ハイパーパラメータを抽出
    alpha = trial.params['alpha']

    #
    # MMMパイプラインの構築&学習
    #

    MMM_pipeline = build_MMM_pipeline_regarima(
        X, y, 
        apply_effects_features, 
        carryover_params, 
        curve_params,
        alpha)

    # パイプラインを使って学習
    trained_model, pred = train_and_evaluate_regarima(MMM_pipeline, X, y)

    # 最適ハイパーパラメータの集約
    model_params = [
        carryover_params, 
        curve_params, 
        alpha]

    return trained_model,model_params,pred

# 
# 実測値と予測値の時系列推移
#

def plot_actual_vs_predicted(index, actual_values, predicted_values, ylim):
    """
    実際の値と予測値を比較するグラフを描画する関数。

    :param index: データのインデックス
    :param actual_values: 実際の値の配列
    :param predicted_values: 予測値の配列
    :param ylim: y軸の表示範囲
    """
    fig, ax = plt.subplots()

    ax.plot(index, actual_values, label="actual")
    ax.plot(index, predicted_values, label="predicted", linestyle="dotted", lw=2)

    ax.set_ylim(ylim)

    plt.title('Time series of actual and predicted values')
    plt.legend()
    plt.show()

#
# 貢献度の算出
#

def calculate_and_plot_contribution(y, X, model, ylim=None, apply_effects_features=None):
    """
    各媒体の売上貢献度を算定し、結果をプロットする関数。

    :param y: ターゲット変数
    :param X: 特徴量のデータフレーム
    :param model: 学習済みモデル
    :param ylim: y軸の表示範囲
    :return: 各媒体の貢献度
    """
    if apply_effects_features is None:
        apply_effects_features = X.columns

    # yの予測
    pred = model.predict(X)
    pred = pd.DataFrame(pred, index=X.index, columns=['y'])

    # 値がすべて0の説明変数
    X_ = X.copy()
    X_[apply_effects_features] = 0

    # Baseの予測
    base = model.predict(X_)
    pred['Base'] = base

    # 各媒体の予測
    for feature in apply_effects_features:
        X_[apply_effects_features] = 0
        X_[feature] = X[feature]
        pred[feature] = np.maximum(model.predict(X_) - base, 0)

    # 予測値の補正
    predy = pred.drop(['y'], axis=1).sum(axis=1)
    correction_factor = y.div(predy, axis=0)
    pred_adj = pred.mul(correction_factor, axis=0)
    contribution = pred_adj.drop(columns=['y'])

    # apply_effects_features以外をbaseに集約
    contribution['Base'] = contribution.drop(columns=apply_effects_features).sum(axis=1)
    contribution = contribution[['Base'] + list(apply_effects_features)]

    # エリアプロット
    if ylim is not None:
        ax = contribution.plot.area()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels))
        ax.set_ylim(ylim)
        plt.title('Contributions over time')
        plt.show()

    return contribution

#
# 貢献度構成比の算出
#

def summarize_and_plot_contribution(contribution):
    """
    媒体別の売上貢献度の合計と構成比を計算し、結果を表示する関数。

    :param contribution: 各媒体の貢献度を含むデータフレーム
    :return: 売上貢献度の合計と構成比を含むデータフレーム
    """
    # 各媒体の貢献度の合計
    contribution_sum = contribution.sum(axis=0)

    # 各媒体の貢献度の構成比
    contribution_percentage = contribution_sum / contribution_sum.sum()

    # 結果を1つのDataFrameにまとめる
    contribution_results = pd.DataFrame({
        'contribution': contribution_sum,
        'ratio': contribution_percentage
    })

    # グラフ化
    contribution_sum.plot.pie()
    plt.title('Contribution Composition Ratio')
    plt.show()

    return contribution_results

#
# マーケティングROIの算出
#

def calculate_marketing_roi(X, contribution):
    """
    各媒体のマーケティングROIを算定する関数。

    :param X: 各媒体のコストを含むデータフレーム
    :param contribution: 各媒体の売上貢献度を含むデータフレーム
    :return: 各媒体のROIを含むデータフレーム
    """
    # 各媒体の貢献度の合計
    contribution_sum = contribution.sum(axis=0)
    
    # 各媒体の貢献度の構成比
    contribution_percentage = contribution_sum / contribution_sum.sum()   
    
    # 各媒体のコストの合計
    cost_sum = X.sum(axis=0)

    # 各媒体のROIの計算
    ROI = (contribution[X.columns].sum(axis=0) - cost_sum)/cost_sum

    # グラフ
    ROI.plot.bar()
    plt.title('Marketing ROI')
    plt.show()

    return ROI

#
# 散布図作成（売上貢献度×マーケティングROI）
#

def plot_scatter_of_contribution_and_roi(X, contribution):
    """
    売上貢献度とマーケティングROIの散布図を作成する関数。

    :param contribution_results: 各媒体の売上貢献度を含むデータフレーム
    :param ROI: 各媒体のROIを含むデータフレーム
    :param cost: 各媒体のコストを含むシリーズ
    """

    # 各媒体の貢献度の合計
    contribution_sum = contribution.sum(axis=0)

    # 各媒体の貢献度の構成比
    contribution_percentage = contribution_sum / contribution_sum.sum()

    # 各媒体のコストの合計
    cost_sum = X.sum(axis=0)

    # 各媒体のROIの計算
    ROI = (contribution[X.columns].sum(axis=0) - cost_sum)/cost_sum
    
    # データフレーム作成
    data_to_plot = pd.DataFrame({
        'contribution_percentage': contribution_percentage,
        'ROI': ROI,
        'cost': cost_sum
    })
    data_to_plot = data_to_plot.dropna()

    # 散布図作成
    plt.scatter(
        data_to_plot.contribution_percentage, 
        data_to_plot.ROI, 
        s=data_to_plot.cost/data_to_plot.cost.sum()*10000,
        alpha=0.5
    )
    
    # ラベ゙ル付け
    for i, row in data_to_plot.iterrows():
        plt.text(row.contribution_percentage, row.ROI, i)

    # 横軸と縦軸のスケールを1.2倍づつ左右上下増やす
    plt.xlim([
        0, 
        max(data_to_plot.contribution_percentage)*1.2])
    plt.ylim([
        -max(abs(data_to_plot.ROI))*1.2,
        max(abs(data_to_plot.ROI))*1.2
        ])

    # タイトルや軸ラベルなど
    plt.xlabel('Contribution Percentage')
    plt.ylabel('Marketing ROI')
    plt.title('Scatter plot of Media Channels')
    plt.grid(True)
    plt.show()

    return data_to_plot

#
# 学習済みのRegARIMAモデルの予測と評価
#

def test_evaluate_regarima(model, X, y, term):
    """
    学習済みのモデルの予測と評価を行う関数
    引数:
    - model: 学習済みのRegARIMAモデルのインスタンス
    - X: テストデータの特徴量データフレーム
    - y: テストデータのターゲット変数
    - term: 予測対象期間（例: 52週）
    戻り値:
    - 予測値pred（テストデータ期間）
    """

    # 予測
    pred = model.predict(X, n_periods=term)
    pred = pd.DataFrame(pred, index=X.index, columns=['y'])

    # 精度指標の計算
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    mape = mean_absolute_percentage_error(y, pred)
    r2 = r2_score(y, pred)

    # 精度指標の出力
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('MAPE:', mape)
    print('R2:', r2)

    # 実測値と予測値の時系列推移
    plot_actual_vs_predicted(y.index, y.values, pred.y.values, (0, np.max(y) * 1.1))

    return pred

##########################
# 最適投資配分
##########################

#
# 最適投資配分
#

def optimize_investment(trained_model, X_actual, optimized_features, cost_all, niter = 100):
    """
    与えられた機械学習モデルと特徴量のデータフレームを受け取り、最適な広告出稿配分を算出する。
    
    引数:
    - trained_model: 学習済みの機械学習モデル
    - X_actual: 特徴量のデータフレーム
    - optimized_features: 最適化対象の特徴量（アドストックを考慮しない特徴量）
    - niter: int, 最適化プロセスにおける反復回数。デフォルトは100
    
    戻り値:
    - opt_results: {
        'optimizeResult': 最適化の結果
        'X_optimized': 最適化期間における対象特徴量のデータフレーム
    }
    """

    # 最適化する期間
    term = X_actual.shape[0]

    # プログレスバーを表示するためのインスタンス
    progression = tqdm.tqdm(
        total=niter,
        desc="Optimization",
        position=0, 
        leave=True,
        dynamic_ncols=True) 

    # 最適化期間における対象特徴量のデータを抽出
    X_before = X_actual[optimized_features].copy()

    # 最適化期間における対象外特徴量のデータを抽出
    X_no_effects = X_actual.drop(optimized_features, axis=1).copy()

    # 目的関数の定義
    def objective_function(x):
        # xをデータフレームに変換
        x_pd = pd.DataFrame(
            x.reshape(term, len(X_before.columns)),
            index=X_before.index, 
            columns=X_before.columns)
        # 最適化対象外の特徴量との結合
        x_combined = pd.concat(
            [x_pd, X_no_effects], 
            axis=1)
        x_combined = x_combined[X_actual.columns]
        # 予測値の計算
        pred_value = trained_model.predict(x_combined, n_periods=term)
        # 目的関数の値（予測値の和の負値）を返す
        return -1 * pred_value.sum()

    # コスト全体の設定
    cost_all = cost_all

    # 制約条件の定義：総投資額が同じになるように
    constraint = {"type":"eq","fun": lambda x: cost_all-x.sum()}

    # 投資額の上限と下限の定義
    lower = [0] * X_before.size
    upper = [np.inf] * X_before.size

    # minimizer_kwargsの設定
    minimizer_kwargs = dict(
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=(constraint))

    # 初期解の設定
    x_0 = X_before.values.ravel()

    # 初期解の目的関数値の合計の負値
    init_obj_value = -1 * objective_function(x_0)

    # 目的関数の再定義（初期解より改善）
    def revised_objective_function(x):
        obj_value = objective_function(x)
        return obj_value if obj_value < init_obj_value else 0

    # コールバック関数の定義
    best_value = 0
    def callback(xk, f, accept):
        nonlocal best_value
        if f < best_value:
            best_value = f
            progression.set_description(f"Optimal value: {best_value:.2f}")
        progression.update(1)

    # 最適化の実行
    optimizeResult = basinhopping(
        revised_objective_function,
        x_0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter, 
        T=1000, 
        stepsize=1e5,
        seed=0,
        callback=callback)

    # プログレスバーを閉じる
    progression.close()

    # 最適解と最適化対象外の特徴量を結合
    x_pd = pd.DataFrame(
        optimizeResult.x.reshape(term, len(X_before.columns)),
        index=X_before.index, 
        columns=X_before.columns)
    X_optimized = pd.concat(
        [x_pd, X_no_effects], 
        axis=1)
    X_optimized = X_optimized[X_actual.columns]

    # 最適化の結果を返す
    opt_results = {
        'optimizeResult':optimizeResult, 
        'X_optimized':X_optimized
    }
    return opt_results

#
# 最適解の取得
#

def get_optimized_allocation(opt_results):
    """
    最適投資配分を計算して返す関数

    引数:
    - results: {
        'optimizeResult': OptimizeResult, 最適化の結果
        'X_before': DataFrame, 最適化する特徴量（アドストックを考慮する特徴量）の直近term間のデータ
        'X_no_effects': DataFrame, 最適化しない特徴量（アドストックを考慮しない特徴量）の直近term間のデータ
    }

    戻り値:
    - X_optimized (pd.DataFrame): 最適化後の特徴量データ（アドストックを考慮しない特徴量も含む）
    """

    # 最適化の結果を取得
    optimizeResult = opt_results['optimizeResult']
    X_before = opt_results['X_before']
    X_no_effects = opt_results['X_no_effects']

    # 予測結果から最適化後の値をDataFrameに変換
    X_after = pd.DataFrame(
        optimizeResult.x.reshape(len(X_before), len(X_before.columns)),
        index=X_before.index,
        columns=X_before.columns)

    # 最適化後のDataFrameと変更しない特徴量を結合
    X_optimized = pd.concat(
        [X_after, X_no_effects], 
        axis=1)
    
    return X_optimized

#
# 最適化前と後の貢献度推移プロット
#

def opt_calculate_and_plot_contribution(X_optimized, X_actual, y_actual, trained_model, ylim):
    """
    各媒体の売上貢献度を算定し、最適化前と後の結果をプロットする関数。

    引数:
    - X_actual (pd.DataFrame): 実測値の特徴量のデータフレーム
    - X_optimized (pd.DataFrame): 最適化後の特徴量のデータフレーム
    - y_actual (pd.Series): 実測値のターゲット変数
    - trained_model: 学習済みモデル
    - ylim (tuple): y軸の表示範囲

    戻り値:
    - contributions: {
        0:'Contributions before Optimization',pd.DataFrame, 最適化前の貢献度
        1:'Contributions after Optimization',pd.DataFrame, 最適化後の貢献度
    }
    """

    contributions = {}
    titles = ['Contributions before Optimization', 'Contributions after Optimization']

    fig, ax = plt.subplots(2, 1)

    for subplot_index, (X, title) in enumerate(zip([X_actual, X_optimized], titles)):

        # yの予測
        pred = pd.DataFrame(trained_model.predict(X), index=X.index, columns=['y'])

        # 値がすべて0の説明変数
        X_ = X.copy()
        X_.iloc[:, :] = 0

        # Baseの予測
        base = trained_model.predict(X_)
        pred['Base'] = base

        # 各媒体の予測
        for i in range(len(X.columns)):
            X_.iloc[:, :] = 0
            X_.iloc[:, i] = X.iloc[:, i]
            pred[X.columns[i]] = trained_model.predict(X_) - base

        # 予測値の補正
        if subplot_index == 0:
            correction_factor = y_actual.div(pred.y, axis=0)
            pred_adj = pred.mul(correction_factor, axis=0)
            contribution = pred_adj.drop(columns=['y'])
        else:
            contribution = pred.drop(columns=['y'])

        # 貢献度の格納
        contributions[subplot_index] = contribution

        # エリアプロット
        ax[subplot_index].stackplot(contribution.index, contribution.T)
        ax[subplot_index].legend(contribution.columns)
        ax[subplot_index].set_ylim(ylim)
        ax[subplot_index].set_title(title)

    plt.tight_layout()
    plt.show()

    return contributions

#
# 現状と最適配分時の比較（yとマーケティングROI）
#

def compare_y_and_marketing_roi(X_optimized, X_actual, y_actual, trained_model, apply_effects_features):
    """
    現状と最適配分時のyおよびマーケティングROIを比較し、結果をグラフとして表示します。
    戻り値として、実測値と最適配分時のyの合計やマーケティングROIなどを含む辞書を返します。

    引数:
    - X_optimized (pd.DataFrame): 最適化後の特徴量
    - X_actual (pd.DataFrame): 実測値の特徴量
    - y_actual (pd.Series): 実測値のターゲット変数
    - trained_model: 学習済みの機械学習モデル
    - apply_effects_features (list): 適用する特徴量のリスト

    戻り値:
    - dict: {
        'y_actual': 実測値のyの合計,
        'y_optimized': 最適配分時のyの合計,
        'y_change_percent': yの変化率（パーセント）,
        'roi_actual': 実測値のマーケティングROI,
        'roi_optimized': 最適配分時のマーケティングROI,
        'roi_change_point': ROIの変化ポイント
    }
    """

    # 元の総投資額の計算
    cost_all = X_actual[apply_effects_features].sum().sum()

    # yの実測値の合計
    y_actual_sum = round(y_actual.sum())

    # 修正比率の計算（実測のyと予測の比）
    correction_ratio = y_actual.sum()/trained_model.predict(X_actual).sum()

    # 最適配分時のyの合計
    pred = trained_model.predict(X_optimized)
    y_optimized_sum = round(pred.sum()*correction_ratio)

    # yの変化率の計算
    y_change_percent = ((y_optimized_sum - y_actual_sum) / y_actual_sum) * 100

    # yの変化率の符号
    sign_y = '+' if y_change_percent >= 0 else '-'

    # 現状のマーケティングROI
    roi_actual = ((y_actual_sum - cost_all) / cost_all) 

    # 最適配分時のマーケティングROI
    roi_optimized = ((y_optimized_sum - cost_all) / cost_all) 

    # roiの変化ポイントの計算
    roi_change_point = roi_optimized - roi_actual

    # roiの変化の符号
    sign_roi = '+' if roi_change_point >= 0 else '-'

    # グラフのFigure
    fig, axs = plt.subplots(2, 1, figsize=(12,14))
    
    labels = ['Actual', 'Optimized']
    y_values = [y_actual_sum, y_optimized_sum]
    roi_values = [roi_actual, roi_optimized]

    axs[0].bar(labels, y_values, color=['blue', 'orange'])
    axs[0].set_title('Comparison of y: Actual vs. Optimized')
    axs[0].set_ylabel('y')
    for i, value in enumerate(y_values):
        axs[0].text(i, value, f'{int(value)}', ha='center', va='bottom')

    axs[1].bar(labels, roi_values, color=['blue', 'orange'])
    axs[1].set_title('Comparison of Marketing ROI: Actual vs. Optimized')
    axs[1].set_ylabel('ROI')
    for i, value in enumerate(roi_values):
        axs[1].text(i, value, f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return {
        'y_actual_sum': f"{y_actual_sum}",
        'y_optimized_sum': f"{y_optimized_sum}",
        'y_change_percent': f"{sign_y}{abs(y_change_percent):.2f} %",
        'roi_actual': f"{roi_actual:.2f}",
        'roi_optimized': f"{roi_optimized:.2f}",
        'roi_change_point': f"{sign_roi}{abs(roi_change_point):.2f} points"
    }

#
# 投資配分構成比の比較
#

def plot_comparative_allocation(X_actual, X_optimized, apply_effects_features):
    """
    最適化前と最適化後のメディアコスト配分の構成比を比較し、グラフで表示する関数。

    引数:
    - X_actual (pd.DataFrame): 実測値の特徴量
    - X_optimized (pd.DataFrame): 最適化後の特徴量
    - apply_effects_features (list): 適用する特徴量のリスト（コストデータの特徴量）

    戻り値:
    - comparison_df (pd.DataFrame): 最適化前と最適化後のメディアコストの割合を比較したDataFrame

    """

    # 最適化前（実投資）の構成比
    X_before = X_actual[apply_effects_features]
    X_before_ratio = X_before.sum()/X_before.sum().sum()

    # 最適投資の構成比
    X_after = X_optimized[apply_effects_features]
    X_after_ratio = X_after.sum()/X_after.sum().sum()

    # 構成比の表示
    comparison_df = pd.DataFrame({
        'Actual Allocation': X_before_ratio, 
        'Optimized Allocation': X_after_ratio})

    # グラフ化
    fig, ax = plt.subplots()

    bottom = np.zeros(2)
    for value, feature in zip(comparison_df.values, apply_effects_features):
        ax.bar(['Actual', 'Optimized'], value, bottom=bottom, label=feature)
        bottom += value

    ax.set_title('Media Cost Allocation')
    ax.set_ylabel('% of Total Investment')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.set_ylim([0, 1]) 

    plt.show()

    return comparison_df

    
