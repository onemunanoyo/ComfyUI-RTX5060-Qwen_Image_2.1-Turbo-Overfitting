# ComfyUI RTX 5060 / Qwen Image 2.1 Turbo

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) をベースに、**RTX 5060 8GB・Ryzen 7 9700X・RAM 32GB** で Qwen Image 2.1 Turbo の推論を調整する実験用forkです。

リポジトリ名の「Overfitting」は、特定のPCとワークロードに合わせた推論側の調整を指します。モデルの再学習や追加の重み量子化は行っていません。公式の説明は [README.upstream.md](README.upstream.md)、実験の詳細は [experiments/rtx5060/README.md](experiments/rtx5060/README.md) にあります。

## 実測結果

**同じQ6_Kで約18.0秒 → 約11.8秒／枚、約35%短縮。**

| 実行設定 | 1枚あたり |
| --- | ---: |
| 通常BF16・PyTorch/cuDNN Attention | 17.99秒 |
| FP16計算＋FP16高速累積 | 12.70秒 |
| FP16高速累積＋Comfy Kitchen Attention | **11.76秒** |

条件は1024×1024、batch 1、Euler、CFG 1、Turbo専用sigma列、8ステップ。プロンプトを固定してseedを変え、初回ロードを除外しています。VAEデコードとPNG保存を含む実時間です。最終比較のサンプル数は通常2枚・高速2枚で、高速条件は別の3枚でも11.75〜11.81秒でした。[測定条件とサンプル値](experiments/rtx5060/benchmark-summary.json)

高速設定では演算精度が変わります。Kitchen Attentionは内部でINT8演算を使い、同じseedでもBF16版とは画像の細部が変わります。確認したのは風景プロンプトと4つのseedで、顔・文字・画像編集・別解像度での品質や速度は未評価です。性能値はこのPCでの結果です。

## このforkの変更

- 動的ロードのテキストエンコーダーからGGUFへ切り替える際、既存のRAMキャッシュを維持する小さな修正。RAM圧迫時の解放と明示的な全解放は維持します。
- ComfyUI-GGUF用パッチ：Qwen Image 2.1の認識と、Q6_K/Q8_0の融合展開カーネル。
- FP16高速累積を選べる独立した起動スクリプト。
- 実験用スクリプト、CPUで動くメモリ管理の回帰テスト、公開用の測定サマリー。

`torch.compile`は定常速度の利点が小さく、モデル入れ替え後の再コンパイルも発生したため、標準の高速起動には含めていません。Animaは従来どおり約4.8秒で、採用できる改善は確認していません。

## セットアップ（検証環境：Windows）

既存のportable版とは別のフォルダでセットアップします。以下はリポジトリのルートで実行してください。Python 3.13と[uv](https://docs.astral.sh/uv/)を使用します。

```powershell
git clone --branch codex/rtx5060-inference https://github.com/onemunanoyo/ComfyUI-RTX5060-Qwen_Image_2.1-Turbo-Overfitting.git
cd ComfyUI-RTX5060-Qwen_Image_2.1-Turbo-Overfitting
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/rtx5060/requirements.txt --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes/ComfyUI-GGUF
git -C custom_nodes/ComfyUI-GGUF checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
git -C custom_nodes/ComfyUI-GGUF apply ../../experiments/rtx5060/patches/gguf-rtx5060.patch
```

このパッチは上記のGGUFコミット向けです。任意の最新版への適用は未検証です。検証環境のPyTorchは2.14.1+cu130、ドライバーは617.42です。

### モデルの配置

モデルは同梱していません。必要なファイルを各配布元から用意し、ライセンス・利用条件を確認してください。

| ファイル | 標準の配置先 |
| --- | --- |
| `Qwen-Image-2.1-Turbo-AD-Q6_K.gguf` | `models/diffusion_models/` |
| `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |

参考：[AtomicChat Turbo GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF)、[Comfy-Org Qwen Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。既存のモデルを参照する場合は、標準の`extra_model_paths.yaml.example`を参考にローカルの`extra_model_paths.yaml`を作成します。

### 起動

```powershell
# 通常BF16
.\run_experiment.ps1
# FP16高速累積（約12.7秒）
.\run_experiment.ps1 -QwenFast
# FP16高速累積＋Kitchen Attention（約11.8秒）
.\run_experiment.ps1 -QwenFast -ExtraArgs @('--use-ck-attention')
```

UIは http://127.0.0.1:8190 。モードを切り替えるときはCtrl+Cで停止してから起動します。`-Port 8191`のようにポートを変更できます。起動スクリプトはlocalhost限定・offline設定で実行します。`-QwenFast`はQwen向けの実験設定で、Animaなどは通常起動で使ってください。

## 8ステップでの再計測

```powershell
.venv/Scripts/python.exe experiments/rtx5060/tools/benchmark_local.py --model qwen --runs 4 --sigmas '1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0' --output experiments/rtx5060/results/qwen.json
```

`--vary-prompt`を追加すると毎回プロンプトが変わります。初回読み込み、プロンプト固定、プロンプト変更ありは別々に評価してください。古いEuler/simpleの実験結果は専用sigma列の結果と区別して記録しています。

## 検証・構成

```powershell
uv pip install --python .venv/Scripts/python.exe -r tests-unit/requirements.txt
.venv/Scripts/python.exe -m pytest tests-unit/comfy_test/test_mixed_model_offload.py -q
```

実験のコード・パッチは`experiments/rtx5060/`、ローカルの計測結果・ログ・プロファイルはその配下の`results/`・`logs/`・`profiles/`、生成画像は`output/benchmark/`です。モデル、Python環境、キャッシュ、ローカル設定、生成画像、未整理の実行記録はGitに含めません。

## 出典・ライセンス

ComfyUI本体は[公式リポジトリ](https://github.com/Comfy-Org/ComfyUI)を基にしており、[GPL-3.0](LICENSE)を維持しています。GGUFノードのパッチ元は[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)（Apache-2.0）です。追加のTritonカーネルもApache-2.0として配布します。モデルのライセンスは各配布元に従います。

このfork固有の不具合は[このリポジトリのIssues](https://github.com/onemunanoyo/ComfyUI-RTX5060-Qwen_Image_2.1-Turbo-Overfitting/issues)へお願いします。
