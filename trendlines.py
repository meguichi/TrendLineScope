import numpy as np
import pandas as pd
import scipy.signal as sig
from sklearn.linear_model import RANSACRegressor


def detect_pivots(df: pd.DataFrame, order: int = 5):
    """
    price の局所的な極大・極小点（ピボット）を検出。
    order: 前後何本を比較して極値とみなすか
    """
    highs = sig.argrelextrema(df['high'].values, np.greater, order=order)[0]
    lows = sig.argrelextrema(df['low'].values, np.less, order=order)[0]
    return highs, lows


def fit_trendlines(df: pd.DataFrame,
                   idxs: np.ndarray,
                   kind: str = 'resistance',
                   residual_threshold: float = 1.0,
                   min_samples: int = 2):
    """
    ピボット同士を線形回帰＋RANSACでフィットし、
    inlier 比率が高いラインを返す。

    kind: 'resistance'→highs で引く／'support'→lows で引く
    residual_threshold: RANSAC の残差許容値
    min_samples: RANSAC の最小サンプル数
    """
    lines = []
    X_full = np.arange(len(df)).reshape(-1, 1)
    y_full = df['high'].values if kind == 'resistance' else df['low'].values

    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            xi, xj = idxs[i], idxs[j]
            pts_x = np.array([xi, xj]).reshape(-1, 1)
            pts_y = y_full[[xi, xj]]

            # 2点で初期モデル -> RANSAC
            model = RANSACRegressor(residual_threshold=residual_threshold,
                                    min_samples=min_samples,
                                    random_state=0)
            model.fit(pts_x, pts_y)
            inliers = model.inlier_mask_
            # inlier の占める割合が 50% 以上なら採用
            if inliers.mean() >= 0.5:
                slope = model.estimator_.coef_[0]
                intercept = model.estimator_.intercept_
                lines.append((slope, intercept))
    return lines