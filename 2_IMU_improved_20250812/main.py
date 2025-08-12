"""
IMU改良モデル - メイン実行スクリプト
訓練から推論まで包括的に実行
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

warnings.filterwarnings("ignore")


def main():
    """メイン実行関数"""

    parser = argparse.ArgumentParser(description="IMU改良モデル訓練スクリプト")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference", "both"],
        help="実行モード: train, inference, or both",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="推論時に使用するモデルディレクトリ"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" " * 15 + "🚀 IMU改良モデル v2.0.0 🚀")
    print("=" * 70)
    print(f"実行モード: {args.mode}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    if args.mode in ["train", "both"]:
        print("\n📊 モデル訓練を開始します...")
        print("-" * 70)

        try:
            from src.train_model import main as train_main

            # 訓練を実行
            save_dir = train_main()

            print("\n✅ モデル訓練が完了しました!")
            print(f"保存先: {save_dir}")

            # 推論モードの場合、訓練したモデルを使用
            if args.mode == "both":
                args.model_dir = str(save_dir)

        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    if args.mode in ["inference", "both"]:
        print("\n🔮 推論を開始します...")
        print("-" * 70)

        if args.model_dir is None and args.mode == "inference":
            print("❌ エラー: 推論モードではmodel_dirを指定してください")
            sys.exit(1)

        try:
            from src.inference import run_inference

            # 推論を実行
            submission_path = run_inference(args.model_dir)

            print("\n✅ 推論が完了しました!")
            print(f"提出ファイル: {submission_path}")

        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 70)
    print(" " * 20 + "🎉 処理完了 🎉")
    print("=" * 70)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
