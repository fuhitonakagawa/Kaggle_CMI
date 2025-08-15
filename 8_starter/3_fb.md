丁寧に実装されています。**IMU の“ワールド化＋線形加速度＋角速度＋周波数＋変化点（ピーク）＋自己相関＋多スケール要約（ピラミッド/テール）”**まで一通り入っており、学習/推論のパリティも概ね担保できています。
そのうえで、精度・安定性・保守性の観点から**要チェック/改善ポイント**を“コードに直結する形”で指摘します。最後に**IMU のみ / マルチモダリティの両ケースでの挙動**も整理しました。

---

## ✅ よい点（そのままでも OK）

- **学習/推論のパリティ**

  - 同名の `extract_features` を学習・推論で共通化し、推論側は `feature_names` に合わせて欠落列を 0 埋め → スキーマ整合 OK。
  - CV は **StratifiedGroupKFold(被験者グループ化)** でリーク防止。fold ごとの**重み付き確率平均**で推論 → 安定性に寄与。

- **IMU 前処理**

  - 四元数の正規化/符号連続化、ワールド座標化、線形加速度、角速度の導出まで網羅。
  - Welch-PSD、バンドパワー、スペクトル重心/ロールオフ/エントロピー、ZCR、ピーク、自己相関、ピラミッド/テール など特徴の被りが少なく網羅的。

- **ロバストな例外処理**

  - 失敗時の 0 代入、NaN の 0 置換が行き届いており、**推論クラッシュしにくい**設計。

---

## ⚠️ 要修正（バグ/仕様ズレ）

1. **テール窓の実装がドキュメントと不一致**

```python
tail_size = max(int(len(data) * tail_fraction), min(20, len(data)))
```

これだと **len≥20 では常に 20 サンプル**になり、“20% のテール”になりません。
→ **修正案（定義通り 20%・下限を少数サンプルに合わせて）**

```python
tail_size = max(int(round(len(data) * tail_fraction)), 1)
tail_size = min(tail_size, len(data))
```

2. **周波数帯 `8–12 Hz` がナイキストを超える可能性**（fs=20Hz なら Nyquist=10Hz）
   → **バンド上限は必ず `min(high, fs/2)` にクランプ**してください。

```python
high_eff = min(high, fs/2.0)
band_mask = (f >= low) & (f <= high_eff)
```

3. **ZCR を「大きさ（magnitude）」系列に適用**
   `acc_magnitude` や `angular_vel_magnitude` は非負なので ZCR は意味が薄い（≈0）。
   → **magnitude 系の ZCR は除外**し、**軸別信号**に対してのみ計算に限定するのが無難です。

4. **角速度/ジャークの `dt` が固定値（20Hz 前提）**
   データのサンプルレートが可変/揺れる場合、速度・周波数特徴が歪みます。
   → **シーケンスから `dt` を推定**（列があれば：`timestamp`/`time`/`elapsed_time`/`seconds_elapsed` を探索、なければ 20Hz フォールバック）

```python
def infer_dt(seq_df, default_fs=20.0):
    tcol = next((c for c in ['timestamp','time','elapsed_time','seconds_elapsed'] if c in seq_df.columns), None)
    if tcol is None:
        return 1.0/default_fs, default_fs
    t = np.asarray(seq_df[tcol], dtype=float)
    # 単位補正（ns→s 等）: サンプル差分のメディアンが異常に小さければ 1e9 や 1e3 で割る等の保険を入れてもOK
    dt = np.median(np.diff(t))
    if dt <= 0:  # フォールバック
        dt = 1.0/default_fs
    return dt, 1.0/dt
```

→ `compute_angular_velocity`, `extract_jerk_features`, `extract_frequency_features` に `dt/fs` を伝播してください。

5. **重力除去（線形加速度）の仮定が強すぎる**
   現在は **固定ベクトル `[0,0,9.81]` 差し引き**ですが、わずかなバイアスや姿勢誤差で残留加速度が生じます。
   → **LPF ベース**で重力を推定して差し引く方法に変更を推奨：

```python
def compute_linear_acceleration(world_acc, fs, method='lpf'):
    if method == 'lpf':
        # 0.75 Hz 以下を重力とみなす（要調整）
        wc = 0.75 / (fs / 2.0)
        b, a = signal.butter(2, wc, btype='low')
        g = signal.filtfilt(b, a, world_acc, axis=0)
        return world_acc - g
    elif method == 'median':
        g = np.median(world_acc, axis=0, keepdims=True)
        return world_acc - g
    else:
        return world_acc - np.array([0.0, 0.0, 9.81])
```

→ これで**被験者/姿勢依存のブレ**が減ります。

6. **オイラー角の統計は“循環統計”にする**
   `as_euler` の出力は ±π の境界でラップし、平均/分散が破綻します。
   → **アンラップ** or **sin/cos で集計**に変更：

```python
angles = r.as_euler('xyz', degrees=False)
angles = np.unwrap(angles, axis=0)  # アンラップ
# or: circular mean
for i, name in enumerate(['roll','pitch','yaw']):
    s, c = np.sin(angles[:, i]), np.cos(angles[:, i])
    features[f"euler_{name}_mean_circ"] = np.arctan2(np.mean(s), np.mean(c))
    features[f"euler_{name}_R"] = np.hypot(np.mean(s), np.mean(c))  # 集中度
```

（既存の直線統計は残しても良いですが、循環統計の方が安定です）

7. **ピーク検出の閾値**
   `prominence=0.5*std` はスケールに敏感。**min distance（サンプル）**や**高さ**も条件に入れるとダマが減ります。

```python
dist = max(int(round(0.15*fs)), 1)  # 例: 150ms 間隔
peaks, prop = signal.find_peaks(data, prominence=0.5*np.std(data), distance=dist)
```

---

## 🔧 改善提案（精度/堅牢性をさらに上げる）

- **相関/自己相関の正規化**
  既に NaN→0 対応はされていますが、**ゼロ分散ケースの事前ガード**を入れるとさらに安定。
- **周波数特徴のウィンドウ化**
  全区間 1 本の PSD に加えて、\*\*複数チャンクの PSD の統計（mean/std/max）\*\*も持つと、非定常動作に強くなります（今のピラミッド要約は時系列統計なので、**周波数側のピラミッド**も効果的）。
- **Hjorth/時系列の滑らかさ**
  活動度・可動度・複雑度（Hjorth）を `linear_acc_magnitude` などへ追加（軽量の割に効きやすい）。
- **クラス重み**
  ラベル不均衡が大きい場合、LightGBM の `class_weight='balanced'` やサンプル重み導入を検討。
- **信頼度に応じた後処理**
  `max_prob<τ` で **BFRB 系を優遇/再分配**（あなたが使っていた「BFRB ブースト」の導入）。fold 確率平均の後に小さなロジスティック回帰で**温度スケーリング**も有効です。

---

## 🧪 学習/推論フロー上の確認ポイント

- **学習/推論の `fs/dt` 一貫性**
  4\)の修正後は、学習時に推定した `fs` に依存する特徴が生まれます。**推論でも同様に `dt/fs` を推定**することで一貫性を担保してください（列が無い場合は 20Hz 固定で学習/推論一致）。
- **四元数の正規化タイミング**
  符号連続化前に**正規化 → 符号連続化 → 再正規化**の順がより安定です（現状も最後に正規化しているので大きな問題はなし）。
- **学習/推論での列存在差**
  推論側は `available_acc_cols / available_rot_cols` を見ています。IMU 列が欠けるケースは想定していない前提ですが、\*_回転が欠損（rot\__ 欄なし）のテスト\*\*も一度実施を（現状は rot 不在だと `compute_world_acceleration` 前でコケます。`if len(available_rot_cols)==4` のガードを入れて、回転が無い場合は “world=dev” にフォールバックするなどの保険を推奨）。

---

## 📦「IMU のみ」と「IMU ＋補助（ToF/THM 等）」の両ケースについて

- **現状の実装**は**IMU 専用**です。**テストが IMU のみでも正常に動作**し、（IMU 以外があっても）**無視するためスコアは安定**します。
  ただし **補助センサが存在する場合の“上乗せ”効果は取り込めていません**。

- **両ケースで高精度を維持**するための実務的設計（推奨）：

  1. **2 系統モデル**を保存

     - **IMU-only モデル（いまの実装）**
     - **Multi-modal モデル（IMU ＋ ToF/THM 特徴）**

       - ToF/THM があるトレーニングシーケンスでは空間統計（8×8 重心/モーメント/非対称/リング差分/PCA 再構成誤差など）と、簡易な時系列統計を足す
       - CV/重み付き確率平均は同じ構成で OK

  2. **推論時に“列存在でゲート”**

     - `if ToF/THM列が存在→ multi-modal モデル確率`
     - `else → IMU-only モデル確率`
     - **両方の確率を持てるケース**（列はあるが欠損多い等）は、\*\*メタ平均（固定重み or 小さなロジスティック/GBM）\*\*で融合

  3. **メタ情報に基づく自動重み**

     - シーケンス内の **欠損率/有効画素率**（ToF=-1 比率など）から multi-modal への重みを自動調整（低品質なら IMU-only を強める）。

> こうしておけば、**IMU のみ**の非公開データでも**IMU-only**で確実にスコアを出し、**補助あり**のときは**上乗せ**が効く構成になります。現状コードは“IMU のみでも問題なく推論可能”ですが、“補助ありでの上乗せ”には未対応という結論です。

---

## ✍️ パッチ例（最小変更で差し替え）

下記だけ置き換えれば、上の主要論点の多くを解消できます。

### 1) dt / fs の推定＋伝播

```python
def infer_dt_and_fs(seq_df, default_fs=20.0):
    tcol = next((c for c in ['timestamp','time','elapsed_time','seconds_elapsed'] if c in seq_df.columns), None)
    if tcol is None:
        return 1.0/default_fs, default_fs
    t = np.asarray(seq_df[tcol], dtype=float)
    dt = np.median(np.diff(t))
    # 単位補正（雑だが安全策）
    if dt < 1e-6: dt /= 1e9
    elif dt < 1e-3: dt /= 1e3
    if dt <= 0 or not np.isfinite(dt):
        dt = 1.0/default_fs
    return dt, 1.0/dt
```

→ `extract_features()` 冒頭で `dt, fs = infer_dt_and_fs(seq_df)` を取得し、
`compute_angular_velocity(rot, dt)`, `extract_jerk_features(..., dt=dt)`, `extract_frequency_features(..., fs=fs)` を渡す。

### 2) 線形加速度（重力推定の LPF 化）

```python
linear_acc_data = compute_linear_acceleration(world_acc_data, fs=fs, method='lpf')
```

### 3) バンド上限のクランプ & magnitude ZCR の抑制

```python
def extract_frequency_features(data, fs=20.0, prefix='freq', compute_zcr=True):
    # ...
    for band_name, (low, high) in bands.items():
        high_eff = min(high, fs/2.0)
        # ...
    # ZCR
    if compute_zcr:
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features[f"{prefix}_zcr"] = zero_crossings / len(data)
```

呼び出し側で `compute_zcr=False` を **magnitude 系**に適用：

```python
extract_frequency_features(acc_magnitude, fs, "acc_mag_freq", compute_zcr=False)
# 軸別は True のまま
```

### 4) テール窓の修正

```python
tail_size = max(int(round(len(data) * tail_fraction)), 1)
tail_size = min(tail_size, len(data))
```

### 5) オイラー角の循環統計

```python
angles = np.unwrap(euler_angles, axis=0)
for i, name in enumerate(['roll','pitch','yaw']):
    s, c = np.sin(angles[:, i]), np.cos(angles[:, i])
    features[f"euler_{name}_mean_circ"] = np.arctan2(np.mean(s), np.mean(c))
    features[f"euler_{name}_R"] = np.hypot(np.mean(s), np.mean(c))
```

### 6) rot\_\* が“不在”の場合のフォールバック（保険）

```python
if len(available_rot_cols) < 4:
    rot_data_clean = np.tile(np.array([1.0,0.0,0.0,0.0]), (len(acc_data),1))  # 単位四元数
else:
    # 既存の前処理
```

---

## 🧪 テスト観点（チェックリスト）

- **可変 fs のシーケンス**で学習 → 推論してスコア/分布が崩れない（dt/LPF の導入確認）。
- \*_rot\__ 欄の欠落/NaN だらけ\*\*のシーケンスでも例外なく通る。
- **周波数帯の端**（8–12Hz）が Nyquist を超えるときでも 0 除算無し/指標が安定。
- **テール窓**が長短シーケンスで意図通り（20%）になる。
- **magnitude 系 ZCR**が常時 0 にならない（=除去される）。
- **学習/推論の特徴数一致**（`feature_names` 相違がない）。
- **CV 折重みの合計=1**、fold 数変更時も問題なし。

---

## まとめ

- **現状の実装でも IMU のみのテストには十分強い**構成です。
- 精度/堅牢性をさらに上げるには、上記 \*\*5 つの重点修正（dt/LPF/バンド上限/テール窓/Euler 循環統計）\*\*から着手してください。
- **補助センサが使える場合に上振れを取りにいく**には、**IMU-only と Multi-modal の二本構成＋推論ゲート**が最も安全です（IMU-only の安定性を損ねない）。

必要なら、ToF/THM の最小実装（8×8 空間統計・-1 率・重心/偏心率・左右/上下差・PCA 再構成誤差＋ CV 学習）と、推論ゲート付きの**2 モデル読み込みコード**の雛形まで出します。
