# プロジェクト概要

## 目的
Kaggleコンペティション「CMI - Detect Behavior with Sensor Data」のためのBFRB（Body-Focused Repetitive Behaviors：身体集中反復行動）検知モデルの開発。

## コンペティション詳細
- **タスク**: 手首装着型デバイスのセンサーデータから毛抜きなどのBFRBを検知・分類
- **データ**: IMU（加速度・ジャイロ）、温度、近接センサー等の多変量時系列データ
- **重要**: テストデータの約50%はIMUのみで構成（他センサー欠損）
- **評価指標**: カスタム指標（階層型macro-F1）
  - 「BFRBか否か」の二値F1
  - BFRB対象クラス間のmacro-F1
  - 上記2つを同等に重み付けして総合スコアを算出

## クラス定義
18クラスの分類タスク：
- 0-7: BFRB（対象行動）
- 8-17: 非BFRB（日常動作）

## 主要アプローチ
1. IMU特化モデル（LightGBM、1D-CNN）
2. フルセンサーモデル（全センサー使用）
3. アンサンブル（IMU-onlyとフルセンサーの組み合わせ）

## 既存のノートブック
- `notebooks-IMU/`: IMU特化のベースライン実装
- `notebooks-CNN/`: CNN系のモデル実装
- `notebooks-TopVoted/`: 高評価のノートブック

## データパス
- 訓練データ: `cmi-detect-behavior-with-sensor-data/train.csv`
- テストデータ: `cmi-detect-behavior-with-sensor-data/test.csv`
- 人口統計: `train_demographics.csv`, `test_demographics.csv`