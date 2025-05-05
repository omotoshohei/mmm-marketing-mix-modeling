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
plt.style.use('ggplot')            # graph style
plt.rcParams['figure.figsize'] = [10, 5]  # default figure size
plt.rcParams['font.size'] = 10     # default font size
import japanize_matplotlib         # enable Japanese in plots

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
warnings.simplefilter('ignore')

##########################
# Time‑series Feature Engineering
##########################

# ── Fourier series terms ────────────────────────────────────────────
def add_fourier_terms(df, num, seasonal):
    """
    Add sine/cosine Fourier terms to capture seasonality.

    Parameters
    ----------
    df : pd.DataFrame
        Original data frame.
    num : int
        Number of Fourier bases (i.e., pairs of sine & cosine).
    seasonal : float
        Period of the Fourier transform (e.g., 52.25 for weekly data).

    Returns
    -------
    pd.DataFrame
        Data frame with additional Fourier columns.
    """
    # Create a time counter 't' if it does not yet exist
    if 't' not in df.columns:
        df['t'] = pd.RangeIndex(start=0, stop=len(df))

    # Add sine and cosine columns
    for i in range(1, num + 1):
        df[f'sin_{i}'] = np.sin(i * 2 * np.pi * df.t / seasonal)
        df[f'cos_{i}'] = np.cos(i * 2 * np.pi * df.t / seasonal)

    return df


# ── Interaction terms between seasonality and selected features ─────
def create_seasonal_interaction_terms(df, X_col, features):
    """Create interaction terms between Fourier columns and given feature names."""
    df = pd.DataFrame(df, columns=X_col)
    for feature in features:
        if feature in df.columns:        # proceed only if column exists
            for trig_func in df.filter(regex='^sin_|^cos_').columns:
                df[f'{feature}_{trig_func}'] = df[feature] * df[trig_func]
    return df

##########################
# Ad‑stock (carry‑over & saturation)
##########################

# ── Carry‑over (geometric decay that can rise & fall) ───────────────
def carryover_advanced(X: np.ndarray, length, peak, rate1, rate2, c1, c2):
    """
    Produce an advanced carry‑over series with different rise & decay slopes.
    """
    X = np.append(np.zeros(length - 1), X)
    Ws = np.zeros(length)

    for l in range(length):
        if l < peak - 1:
            W = rate1 ** (abs(l - peak + 1) ** c1)
        else:
            W = rate2 ** (abs(l - peak + 1) ** c2)
        Ws[length - 1 - l] = W

    carryover_X = []
    for i in range(length - 1, len(X)):
        X_array = X[i - length + 1:i + 1]
        Xi = sum(X_array * Ws) / sum(Ws)
        carryover_X.append(Xi)

    return np.array(carryover_X)


class CustomCarryOverTransformer(BaseEstimator, TransformerMixin):
    """Scikit‑learn transformer that applies carry‑over to selected columns."""
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


# ── Saturation curves (diminishing returns) ─────────────────────────
def exponential_function(x, d):
    """Exponential saturation: 1 - exp(-d * x)."""
    return 1 - np.exp(-d * x)


def logistic_function(x, L, k, x0):
    """Logistic saturation curve."""
    return L / (1 + np.exp(-k * (x - x0)))


class CustomSaturationTransformer(BaseEstimator, TransformerMixin):
    """Apply chosen saturation function (logistic or exponential) to columns."""
    def __init__(self, curve_params=None):
        self.curve_params = curve_params if curve_params is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        transformed_X = np.copy(X)
        for i, params in enumerate(self.curve_params):
            func_name = params.pop('saturation_function')
            if func_name == 'logistic':
                transformed_X[:, i] = logistic_function(X[:, i], **params)
            elif func_name == 'exponential':
                transformed_X[:, i] = exponential_function(X[:, i], **params)
            params['saturation_function'] = func_name  # restore for later use
        return transformed_X


# ── Plot helpers for carry‑over & saturation diagnostics ────────────
def plot_carryover_effect(params, feature_name, fig, axes, i):
    """Visualise the normalised carry‑over kernel for one feature."""
    max_length = max(10, params['length'])
    x = np.concatenate(([1], np.zeros(max_length - 1)))
    y = carryover_advanced(x, **params) / carryover_advanced(x, **params).max()
    params_r = {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}
    axes[2 * i].bar(np.arange(1, max_length + 1), y)
    axes[2 * i].set_title(f'Carryover Effect for {feature_name}')
    axes[2 * i].text(0, 1.1, params_r, ha='left', va='top')
    axes[2 * i].set_xlabel('Lag position (1 = week of investment)')
    axes[2 * i].set_ylabel('Normalised weight')
    axes[2 * i].set_xticks(range(len(y)))
    axes[2 * i].set_ylim(0, 1.1)


def plot_saturation_curve(params, feature_name, fig, axes, i):
    """Visualise the chosen saturation curve for one feature."""
    x = np.linspace(-1, 3, 400)
    func_name = params.pop('saturation_function')
    if func_name == 'logistic':
        y = logistic_function(x, **params)
    elif func_name == 'exponential':
        y = exponential_function(x, **params)
    params_r = {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}
    axes[2 * i + 1].plot(x, y, label=feature_name)
    axes[2 * i + 1].set_title(f'Saturation Curve for {feature_name}')
    params['saturation_function'] = func_name
    axes[2 * i + 1].text(-1, max(y) * 1.1, params_r, ha='left', va='top')
    axes[2 * i + 1].set_xlabel('Scaled spend')
    axes[2 * i + 1].set_ylabel('Transformed value')
    axes[2 * i + 1].set_ylim(0, max(y) * 1.1)
    axes[2 * i + 1].set_xlim(-1, 3)


def plot_effects(carryover_params, curve_params, feature_names):
    """Wrapper that draws both carry‑over kernels and saturation curves."""
    fig, axes = plt.subplots(len(feature_names) * 2, 1, figsize=(12, 10 * len(feature_names)))
    for i, params in enumerate(carryover_params):
        plot_carryover_effect(params, feature_names[i], fig, axes, i)
    for i, params in enumerate(curve_params):
        plot_saturation_curve(params, feature_names[i], fig, axes, i)
    plt.tight_layout()
    plt.show()

##########################
# MMM Core
##########################

# ── RegARIMA hybrid model (Ridge + ARIMA on residuals) ──────────────
class RegARIMAModel:
    """
    Two‑step model:
      1. Ridge regression fits X → y.
      2. ARIMA captures residual autocorrelation.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.ridge_model = Ridge(alpha=alpha)
        self.arima_model = None
        self.residuals = None

    def fit(self, X, y):
        """Fit Ridge, then ARIMA on residuals."""
        self.ridge_model.fit(X, y)
        predictions = self.ridge_model.predict(X)
        self.residuals = y - predictions
        self.arima_model = auto_arima(
            self.residuals,
            seasonal=False,
            stepwise=False,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )

    def predict(self, X, n_periods=0):
        """
        Predict with the hybrid model.
        If n_periods > 0, forecast future residuals with ARIMA.
        """
        ridge_pred = self.ridge_model.predict(X)
        if n_periods > 0:
            arima_pred = self.arima_model.predict(n_periods=n_periods)
        else:
            arima_pred = self.arima_model.predict_in_sample()
        return ridge_pred + arima_pred[0]   # assumes n_periods == 1


# ── Optuna objective for RegARIMA with carry‑over + saturation ──────
def regarima_objective(trial, X, y, apply_effects_features):
    """Return RMSE from time‑series CV for a sampled hyper‑param set."""
    carryover_params, curve_params = [], []

    # Map feature names to column indices
    apply_effects_indices = [X.columns.get_loc(col) for col in apply_effects_features]
    no_effects_indices = list(set(range(X.shape[1])) - set(apply_effects_indices))

    # Sample carry‑over and saturation parameters per feature
    for feature in apply_effects_features:
        # carry‑over
        params_carry = {
            'length': trial.suggest_int(f'carryover_length_{feature}', 1, 10),
            'peak':   trial.suggest_int(f'carryover_peak_{feature}', 1, trial.suggest_int(f'carryover_length_{feature}', 1, 10)),
            'rate1':  trial.suggest_float(f'carryover_rate1_{feature}', 0, 1),
            'rate2':  trial.suggest_float(f'carryover_rate2_{feature}', 0, 1),
            'c1':     trial.suggest_float(f'carryover_c1_{feature}', 0, 2),
            'c2':     trial.suggest_float(f'carryover_c2_{feature}', 0, 2),
        }
        carryover_params.append(params_carry)

        # saturation
        sat_func = trial.suggest_categorical(f'saturation_function_{feature}', ['logistic', 'exponential'])
        if sat_func == 'logistic':
            params_curve = {
                'saturation_function': sat_func,
                'L':  trial.suggest_float(f'curve_param_L_{feature}', 0, 10),
                'k':  trial.suggest_float(f'curve_param_k_{feature}', 0, 10),
                'x0': trial.suggest_float(f'curve_param_x0_{feature}', 0, 2),
            }
        else:
            params_curve = {
                'saturation_function': sat_func,
                'd': trial.suggest_float(f'curve_param_d_{feature}', 0, 10),
            }
        curve_params.append(params_curve)

    alpha = trial.suggest_float('alpha', 1e-3, 1e3)

    # Preprocessing pipeline
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

    seasonal_interaction = FunctionTransformer(
        create_seasonal_interaction_terms,
        kw_args={'X_col': X.columns, 'features': apply_effects_features},
        validate=False,
    )

    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('preprocessor', preprocessor),
        ('create_seasonal', seasonal_interaction),
        ('estimator', RegARIMAModel(alpha=alpha))
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse = np.mean([np.sqrt(-score) for score in scores])
    return rmse


# ── Utility to run Optuna optimisation loop ─────────────────────────
def run_optimization(objective, X, y, apply_effects_features, n_trials=1000, study=None):
    """Create or reuse an Optuna Study and search for best hyper‑params."""
    if study is None:
        study = optuna.create_study(direction='minimize')

    objective_with_data = partial(objective, X=X, y=y, apply_effects_features=apply_effects_features)
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)

    # Show best result
    print("Best trial:")
    best = study.best_trial
    print(f"Value: {best.value}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    return study


# ── Training & evaluation helper for RegARIMA pipeline ──────────────
def train_and_evaluate_regarima(model, X, y, Xmax=1, ymax=1):
    """
    Fit the model, forecast in‑sample, plot actual vs. predicted,
    and print error metrics.
    """
    X_scaled, y_scaled = X / Xmax, y / ymax
    model.fit(X_scaled, y_scaled)
    pred = model.predict(X_scaled)
    pred = pd.DataFrame(pred * ymax, index=X.index, columns=['y'])

    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    mape = mean_absolute_percentage_error(y, pred)
    r2   = r2_score(y, pred)

    plot_actual_vs_predicted(y.index, y.values, pred.y.values, (0, np.max(y) * 1.1))
    print('RMSE:', rmse)
    print('MAE :', mae)
    print('MAPE:', mape)
    print('R2  :', r2)
    return model, pred


# ── Build full MMM pipeline given hyper‑params ──────────────────────
def build_MMM_pipeline_regarima(X, y, apply_effects_features, carryover_params, curve_params, alpha=1):
    """
    Assemble the full scikit‑learn Pipeline for MMM with:
      • Min‑Max scaling
      • Carry‑over & saturation transformers
      • Seasonal‑interaction terms
      • RegARIMA estimator
    """
    effect_idx = [X.columns.get_loc(col) for col in apply_effects_features]
    no_effects = list(set(range(X.shape[1])) - set(effect_idx))

    preprocessor = ColumnTransformer(
        transformers=[
            ('effects', Pipeline([
                ('carryover_transformer', CustomCarryOverTransformer(carryover_params=carryover_params)),
                ('saturation_transformer', CustomSaturationTransformer(curve_params=curve_params))
            ]), effect_idx),
            ('no_effects', 'passthrough', no_effects)
        ],
        remainder='drop'
    )

    seasonal_interaction = FunctionTransformer(
        create_seasonal_interaction_terms,
        kw_args={'X_col': X.columns, 'features': apply_effects_features},
        validate=False,
    )

    return Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('preprocessor', preprocessor),
        ('create_seasonal', seasonal_interaction),
        ('estimator', RegARIMAModel(alpha=alpha))
    ])


# ── Re‑build & fit pipeline from an Optuna trial ────────────────────
def create_model_from_trial_regarima(trial, X, y, apply_effects_features):
    """Extract params from a trial, build pipeline, fit, return model & preds."""
    carry_keys = ['length', 'peak', 'rate1', 'rate2', 'c1', 'c2']
    curve_keys_log = ['L', 'k', 'x0']
    curve_keys_exp = ['d']

    def extract(prefix, feat, keys, params):
        return {k: params[f'{prefix}_{k}_{feat}'] for k in keys}

    carryover_params = [extract('carryover', feat, carry_keys, trial.params) for feat in apply_effects_features]

    curve_params = []
    for feat in apply_effects_features:
        func = trial.params[f'saturation_function_{feat}']
        param = {'saturation_function': func}
        if func == 'logistic':
            param.update(extract('curve_param', feat, curve_keys_log, trial.params))
        else:
            param.update(extract('curve_param', feat, curve_keys_exp, trial.params))
        curve_params.append(param)

    alpha = trial.params['alpha']

    MMM_pipeline = build_MMM_pipeline_regarima(
        X, y, apply_effects_features, carryover_params, curve_params, alpha
    )
    trained_model, pred = train_and_evaluate_regarima(MMM_pipeline, X, y)
    model_params = [carryover_params, curve_params, alpha]

    return trained_model, model_params, pred


# ── Plot actual vs. predicted helper ────────────────────────────────
def plot_actual_vs_predicted(index, actual_values, predicted_values, ylim):
    """Draw line chart of actual vs. fitted values."""
    fig, ax = plt.subplots()
    ax.plot(index, actual_values, label="actual")
    ax.plot(index, predicted_values, label="predicted", linestyle="dotted", lw=2)
    ax.set_ylim(ylim)
    plt.title('Time series of actual and predicted values')
    plt.legend()
    plt.show()

##########################
# Contribution / ROI utilities
##########################

def calculate_and_plot_contribution(y, X, model, ylim=None, apply_effects_features=None):
    """
    Decompose model predictions into base + per‑channel contributions.
    Returns a DataFrame holding weekly contribution by channel.
    """
    if apply_effects_features is None:
        apply_effects_features = X.columns

    pred = pd.DataFrame(model.predict(X), index=X.index, columns=['y'])

    X_ = X.copy()
    X_[apply_effects_features] = 0
    base = model.predict(X_)
    pred['Base'] = base

    for feature in apply_effects_features:
        X_[apply_effects_features] = 0
        X_[feature] = X[feature]
        pred[feature] = np.maximum(model.predict(X_) - base, 0)

    predy = pred.drop(['y'], axis=1).sum(axis=1)
    correction_factor = y.div(predy, axis=0)
    pred_adj = pred.mul(correction_factor, axis=0)
    contribution = pred_adj.drop(columns=['y'])

    contribution['Base'] = contribution.drop(columns=apply_effects_features).sum(axis=1)
    contribution = contribution[['Base'] + list(apply_effects_features)]

    if ylim is not None:
        ax = contribution.plot.area()
        h, l = ax.get_legend_handles_labels()
        ax.legend(reversed(h), reversed(l))
        ax.set_ylim(ylim)
        plt.title('Contributions over time')
        plt.show()

    return contribution


def summarize_and_plot_contribution(contribution):
    """
    Aggregate contribution totals and plot channel share as a pie chart.
    """
    contribution_sum = contribution.sum(axis=0)
    contribution_pct = contribution_sum / contribution_sum.sum()

    result = pd.DataFrame({
        'contribution': contribution_sum,
        'ratio': contribution_pct
    })

    contribution_sum.plot.pie()
    plt.title('Contribution Composition Ratio')
    plt.show()
    return result


def calculate_marketing_roi(X, contribution):
    """
    ROI = (Revenue contribution − Cost) / Cost for each channel.
    """
    cost_sum = X.sum(axis=0)
    ROI = (contribution[X.columns].sum(axis=0) - cost_sum) / cost_sum
    ROI.plot.bar()
    plt.title('Marketing ROI')
    plt.show()
    return ROI


def plot_scatter_of_contribution_and_roi(X, contribution):
    """
    Bubble scatter: X = contribution share, Y = ROI, bubble size = cost.
    """
    contrib_sum = contribution.sum(axis=0)
    contrib_pct = contrib_sum / contrib_sum.sum()
    cost_sum = X.sum(axis=0)
    ROI = (contribution[X.columns].sum(axis=0) - cost_sum) / cost_sum

    df_plot = pd.DataFrame({
        'contribution_percentage': contrib_pct,
        'ROI': ROI,
        'cost': cost_sum
    }).dropna()

    plt.scatter(
        df_plot.contribution_percentage,
        df_plot.ROI,
        s=df_plot.cost / df_plot.cost.sum() * 10000,
        alpha=0.5
    )
    for i, row in df_plot.iterrows():
        plt.text(row.contribution_percentage, row.ROI, i)

    plt.xlim([0, df_plot.contribution_percentage.max() * 1.2])
    plt.ylim([-abs(df_plot.ROI).max() * 1.2, abs(df_plot.ROI).max() * 1.2])
    plt.xlabel('Contribution Percentage')
    plt.ylabel('Marketing ROI')
    plt.title('Scatter plot of Media Channels')
    plt.grid(True)
    plt.show()
    return df_plot

##########################
# Out‑of‑sample Evaluation
##########################
def test_evaluate_regarima(model, X, y, term):
    """
    Forecast `term` periods ahead and report error metrics.
    """
    pred = model.predict(X, n_periods=term)
    pred = pd.DataFrame(pred, index=X.index, columns=['y'])

    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    mape = mean_absolute_percentage_error(y, pred)
    r2   = r2_score(y, pred)

    print('RMSE:', rmse)
    print('MAE :', mae)
    print('MAPE:', mape)
    print('R2  :', r2)

    plot_actual_vs_predicted(y.index, y.values, pred.y.values, (0, np.max(y) * 1.1))
    return pred

##########################
# Budget Allocation Optimisation
##########################
def optimize_investment(trained_model, X_actual, optimized_features, cost_all, niter=100):
    """
    Use basin‑hopping to find spend allocation that maximises predicted sales
    while keeping total spend constant (cost_all).
    """
    term = X_actual.shape[0]
    bar = tqdm.tqdm(total=niter, desc="Optimization", position=0, leave=True, dynamic_ncols=True)

    X_before = X_actual[optimized_features].copy()
    X_no_eff = X_actual.drop(optimized_features, axis=1).copy()

    def objective_function(x):
        x_pd = pd.DataFrame(x.reshape(term, len(X_before.columns)), index=X_before.index, columns=X_before.columns)
        x_combined = pd.concat([x_pd, X_no_eff], axis=1).reindex(columns=X_actual.columns)
        return -trained_model.predict(x_combined, n_periods=term).sum()

    constraint = {"type": "eq", "fun": lambda x: cost_all - x.sum()}
    lower = [0] * X_before.size
    upper = [np.inf] * X_before.size
    minimizer_kwargs = dict(method="SLSQP", bounds=Bounds(lower, upper), constraints=(constraint))
    x0 = X_before.values.ravel()
    init_val = -objective_function(x0)

    def revised_objective(x):
        val = objective_function(x)
        return val if val < init_val else 0

    best_value = 0
    def callback(xk, f, accept):
        nonlocal best_value
        if f < best_value:
            best_value = f
            bar.set_description(f"Optimal value: {best_value:.2f}")
        bar.update(1)

    opt_res = basinhopping(
        revised_objective, x0, minimizer_kwargs=minimizer_kwargs,
        niter=niter, T=1000, stepsize=1e5, seed=0, callback=callback
    )
    bar.close()

    x_opt = pd.DataFrame(opt_res.x.reshape(term, len(X_before.columns)), index=X_before.index, columns=X_before.columns)
    X_optimized = pd.concat([x_opt, X_no_eff], axis=1).reindex(columns=X_actual.columns)

    return {'optimizeResult': opt_res, 'X_optimized': X_optimized}


def get_optimized_allocation(opt_results):
    """Return DataFrame containing optimal spend allocation."""
    opt_res = opt_results['optimizeResult']
    X_before = opt_results['X_before']
    X_no_eff = opt_results['X_no_effects']
    X_after = pd.DataFrame(
        opt_res.x.reshape(len(X_before), len(X_before.columns)),
        index=X_before.index, columns=X_before.columns
    )
    return pd.concat([X_after, X_no_eff], axis=1)


def opt_calculate_and_plot_contribution(X_optimized, X_actual, y_actual, trained_model, ylim):
    """
    Compare contribution before vs. after optimisation and plot stacked areas.
    """
    contributions = {}
    titles = ['Contributions before Optimization', 'Contributions after Optimization']
    fig, ax = plt.subplots(2, 1)

    for idx, (X_df, title) in enumerate(zip([X_actual, X_optimized], titles)):
        pred = pd.DataFrame(trained_model.predict(X_df), index=X_df.index, columns=['y'])
        X_zero = X_df.copy(); X_zero.iloc[:, :] = 0
        base = trained_model.predict(X_zero)
        pred['Base'] = base

        for i in range(len(X_df.columns)):
            X_zero.iloc[:, :] = 0
            X_zero.iloc[:, i] = X_df.iloc[:, i]
            pred[X_df.columns[i]] = trained_model.predict(X_zero) - base

        if idx == 0:
            correction = y_actual.div(pred.y, axis=0)
            contribution = pred.mul(correction, axis=0).drop(columns=['y'])
        else:
            contribution = pred.drop(columns=['y'])
        contributions[idx] = contribution

        ax[idx].stackplot(contribution.index, contribution.T)
        ax[idx].legend(contribution.columns)
        ax[idx].set_ylim(ylim)
        ax[idx].set_title(title)

    plt.tight_layout()
    plt.show()
    return contributions


def compare_y_and_marketing_roi(X_optimized, X_actual, y_actual, trained_model, apply_effects_features):
    """
    Plot and return summary stats comparing y & ROI before vs. after optimisation.
    """
    cost_total = X_actual[apply_effects_features].sum().sum()
    y_actual_sum = round(y_actual.sum())
    correction_ratio = y_actual.sum() / trained_model.predict(X_actual).sum()

    y_opt_sum = round(trained_model.predict(X_optimized).sum() * correction_ratio)
    y_pct = ((y_opt_sum - y_actual_sum) / y_actual_sum) * 100
    sign_y = '+' if y_pct >= 0 else '-'

    roi_actual = (y_actual_sum - cost_total) / cost_total
    roi_opt = (y_opt_sum - cost_total) / cost_total
    roi_diff = roi_opt - roi_actual
    sign_roi = '+' if roi_diff >= 0 else '-'

    fig, axs = plt.subplots(2, 1, figsize=(12, 14))
    axs[0].bar(['Actual', 'Optimized'], [y_actual_sum, y_opt_sum], color=['blue', 'orange'])
    axs[0].set_title('Comparison of y: Actual vs. Optimized')
    axs[0].set_ylabel('y')
    for i, val in enumerate([y_actual_sum, y_opt_sum]): axs[0].text(i, val, f'{int(val)}', ha='center', va='bottom')

    axs[1].bar(['Actual', 'Optimized'], [roi_actual, roi_opt], color=['blue', 'orange'])
    axs[1].set_title('Comparison of Marketing ROI: Actual vs. Optimized')
    axs[1].set_ylabel('ROI')
    for i, val in enumerate([roi_actual, roi_opt]): axs[1].text(i, val, f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return {
        'y_actual_sum': f"{y_actual_sum}",
        'y_optimized_sum': f"{y_opt_sum}",
        'y_change_percent': f"{sign_y}{abs(y_pct):.2f} %",
        'roi_actual': f"{roi_actual:.2f}",
        'roi_optimized': f"{roi_opt:.2f}",
        'roi_change_point': f"{sign_roi}{abs(roi_diff):.2f} points"
    }


def plot_comparative_allocation(X_actual, X_optimized, apply_effects_features):
    """
    Compare media‑spend shares before vs. after optimisation and plot stacked bars.
    """
    X_before = X_actual[apply_effects_features]
    X_before_ratio = X_before.sum() / X_before.sum().sum()

    X_after = X_optimized[apply_effects_features]
    X_after_ratio = X_after.sum() / X_after.sum().sum()

    comparison_df = pd.DataFrame({
        'Actual Allocation': X_before_ratio,
        'Optimized Allocation': X_after_ratio
    })

    fig, ax = plt.subplots()
    bottom = np.zeros(2)
    for val, feat in zip(comparison_df.values, apply_effects_features):
        ax.bar(['Actual', 'Optimized'], val, bottom=bottom, label=feat)
        bottom += val

    ax.set_title('Media Cost Allocation')
    ax.set_ylabel('% of Total Investment')
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1])
    ax.set_ylim([0, 1])
    plt.show()
    return comparison_df
