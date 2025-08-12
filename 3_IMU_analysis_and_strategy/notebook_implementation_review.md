# 📊 CMI二段階分類ノートブック実装レビュー

## 📝 エグゼクティブサマリー

**現在のスコア**: 0.709 (Binary F1: 0.942, Macro F1: 0.475)  
**目標スコア**: 0.770+ (Binary F1: 0.94+, Macro F1: 0.60+)

本レビューでは、我々の`3_IMU_two_stage_all_in_one.ipynb`実装を、上位競技ノートブック（LB 0.77-0.82）と比較分析しました。

---

## 🔍 比較対象ノートブック

### トップスコアノートブック
1. **LB 0.82**: `cmi25-imu-thm-tof-tf-blendingmodel-lb-82.ipynb`
   - 20モデルのアンサンブル
   - StratifiedGroupKFold検証
   - Angular Velocity特徴量

2. **LB 0.77**: `lb-0-77-linear-accel-tf-bilstm-gru-attention.ipynb`
   - 深層学習（BiLSTM + GRU + Attention）
   - Gravity Removal実装
   - MixUp データ拡張

3. **IMU Baseline**: `imu-only-baseline-lgbm-using-worldacc.ipynb`
   - World Acceleration特徴
   - LightGBM with GPU
   - 包括的な統計特徴量

---

## 🎯 我々の実装の強み

### 1. ✅ 二段階分類アーキテクチャ
```python
# 我々の実装
class TwoStageClassifier:
    - Stage 1: Binary (BFRB vs Non-BFRB) - F1: 0.942
    - Stage 2A: BFRB 8-class classification
    - Stage 2B: Non-BFRB 10-class classification
```
**優位性**: 高いBinary F1を活かした階層的アプローチ

### 2. ✅ World Acceleration実装
```python
def compute_world_acceleration(acc, rot):
    # 四元数を使用した座標変換
    rot_scipy = rot[:, [1, 2, 3, 0]]
    r = R.from_quat(rot_scipy)
    world_acc = r.apply(acc)
```
**競合比較**: IMU Baselineと同等の実装品質

### 3. ✅ クラスバランシング対策
- SMOTE（利用可能時）
- class_weight='balanced'
- 階層別の最適化

### 4. ✅ 包括的な特徴量エンジニアリング
- 統計特徴（600+特徴量）
- 周波数領域特徴（FFT、PSD）
- 交差相関特徴
- セグメント特徴（時系列3分割）

---

## ⚠️ 実装の弱点と改善機会

### 1. ❌ **深層学習モデルの欠如**
**問題点**: LightGBMのみの使用（トップモデルは全て深層学習）

**改善案**:
```python
# トップモデルのアーキテクチャ
- Residual SE-CNN blocks
- BiLSTM + GRU + Attention
- Two-branch architecture (IMU + TOF/Thermal)
```

### 2. ❌ **Angular Velocity特徴の欠如**
**LB 0.82の実装**:
```python
def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    # 四元数の微分から角速度を計算
    delta_rot = rot_t.inv() * rot_t_plus_dt
    angular_vel = delta_rot.as_rotvec() / time_delta
```
**影響**: 回転運動の動的パターンを捉えられない

### 3. ❌ **Linear Acceleration実装の違い**
**トップモデル**:
```python
# Butterworthフィルタによる重力推定
b, a = signal.butter(3, 0.3, 'low', fs=sample_rate)
gravity = signal.filtfilt(b, a, world_acc)
linear_acc = world_acc - gravity
```
**我々の実装**: 単純な低周波フィルタ使用

### 4. ❌ **アンサンブル戦略の不足**
**LB 0.82**: 20モデルのブレンディング  
**我々の実装**: 5-fold CVのみ

### 5. ❌ **データ拡張の欠如**
**欠けている技術**:
- MixUp (α=0.4)
- Time stretching
- Gaussian noise injection

---

## 🚀 具体的な改善提案

### 優先度高：即効性のある改善（+0.03〜0.05）

#### 1. Angular Velocity特徴の追加
```python
def add_angular_features(df):
    angular_vel = calculate_angular_velocity_from_quat(df[rot_cols])
    df['angular_vel_x'] = angular_vel[:, 0]
    df['angular_vel_y'] = angular_vel[:, 1] 
    df['angular_vel_z'] = angular_vel[:, 2]
    df['angular_vel_mag'] = np.linalg.norm(angular_vel, axis=1)
    df['angular_distance'] = calculate_angular_distance(df[rot_cols])
```

#### 2. Butterworthフィルタによる重力除去の改良
```python
def improved_gravity_removal(acc, fs=20):
    b, a = signal.butter(3, 0.3, 'low', fs=fs)
    gravity = np.zeros_like(acc)
    for i in range(3):
        gravity[:, i] = signal.filtfilt(b, a, acc[:, i])
    return acc - gravity
```

#### 3. 高度な周波数特徴
```python
# Wavelet変換
from pywt import wavedec
coeffs = wavedec(data, 'db4', level=4)

# スペクトログラムの時間変化
f, t, Sxx = signal.spectrogram(data, fs=20)
```

### 優先度中：モデルアーキテクチャの改善（+0.05〜0.08）

#### 4. 深層学習モデルの導入
```python
def build_hybrid_model():
    # 1D-CNN for local pattern extraction
    x = Conv1D(64, 3, activation='relu')(input)
    x = residual_se_block(x)
    
    # BiLSTM for temporal dependencies
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Attention mechanism
    x = attention_layer(x)
    
    # Combine with LightGBM predictions
    ensemble_pred = 0.7 * dl_pred + 0.3 * lgbm_pred
```

#### 5. MixUpデータ拡張
```python
class MixupGenerator:
    def __init__(self, X, y, alpha=0.4):
        # Binary分類用とMulti-class用で別々のalpha値
        self.alpha_binary = 0.2  # より保守的
        self.alpha_multi = 0.4   # より積極的
```

### 優先度低：長期的な改善（+0.02〜0.03）

#### 6. より大規模なアンサンブル
- 異なる初期化での複数モデル
- 異なるアーキテクチャの組み合わせ
- Stacking with meta-learner

#### 7. 階層的特徴学習
```python
# BFRB専用の特徴抽出器
def extract_bfrb_specific_features():
    # Hair pulling特有のパターン
    # Skin manipulation特有のパターン
    # Scratching特有のパターン
```

---

## 📊 期待される改善効果

| 改善項目 | 実装難易度 | 期待効果 | 推定スコア向上 |
|---------|----------|---------|--------------|
| Angular Velocity | 低 | 高 | +0.02 |
| 改良重力除去 | 低 | 中 | +0.01 |
| 高度な周波数特徴 | 中 | 中 | +0.02 |
| 深層学習導入 | 高 | 非常に高 | +0.05 |
| MixUp | 低 | 中 | +0.01 |
| 大規模アンサンブル | 中 | 高 | +0.03 |

**合計期待改善**: +0.14 (0.709 → 0.849)

---

## 🎯 実装ロードマップ

### Phase 1 (1-2日): 即効性改善
1. Angular Velocity特徴追加
2. Butterworthフィルタ実装
3. 追加の周波数特徴

**期待スコア**: 0.709 → 0.740

### Phase 2 (3-4日): モデル強化
4. 簡易的な1D-CNN追加
5. MixUpデータ拡張
6. アンサンブル拡張（10モデル）

**期待スコア**: 0.740 → 0.770

### Phase 3 (5-7日): 深層学習統合
7. BiLSTM + Attention実装
8. Two-branch architecture
9. 最終アンサンブル最適化

**期待スコア**: 0.770 → 0.800+

---

## 💡 重要な洞察

### 成功要因の分析
1. **トップモデルの共通点**:
   - 全て深層学習ベース
   - 物理的意味のある特徴量（重力除去、角速度）
   - 大規模アンサンブル（10-20モデル）

2. **我々の強み活用**:
   - Binary F1の高さ（0.942）は維持可能
   - 二段階アプローチは理論的に正しい
   - 実装品質は高い

3. **最も重要な改善点**:
   - **深層学習の導入が必須**（全トップモデルが使用）
   - Angular Velocity特徴は即座に追加すべき
   - アンサンブルサイズの拡大

---

## 📝 結論

我々の`3_IMU_two_stage_all_in_one.ipynb`は、堅実な二段階分類アプローチと包括的な特徴量エンジニアリングを実装していますが、トップモデルとの主な差は：

1. **深層学習の欠如**（最重要）
2. **Angular Velocity特徴の欠如**
3. **アンサンブル規模の不足**

これらの改善により、目標スコア0.770は十分達成可能であり、適切な実装により0.800+も視野に入ります。

---

*レビュー実施日: 2025-01-13*  
*レビュー担当: CMI Competition Team*