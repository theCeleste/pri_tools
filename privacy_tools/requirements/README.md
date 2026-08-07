# 方法环境说明

不建议把六套研究代码安装到同一个 Python 环境。原始 requirements 中存在明显冲突，例如 PyTorch、Transformers、Datasets、NumPy 和 Opacus 版本不同。

文件说明：

- `integration.txt`：集成层，无第三方依赖；
- `cliptext-original.txt`：复制自 `ClipText/requirements.txt`；
- `kal_dp-original.txt`：Kal-DP 原始环境依赖快照；
- `zo_dpga-original.txt`：复制自 `ZO-DPGA/requirements.txt`；
- `dadp_icl-original.txt`：DADP-ICL 原始环境依赖快照；
- `privsv-original.txt`：PrivSV 原始环境依赖快照；
- `ptcds-inferred.txt`：根据 PTCDS 入口导入与历史环境快照整理，未锁版本。

推荐做法：

```bash
conda create -n kal_dp python=3.10 -y
conda activate kal_dp
pip install -r privacy_tools/requirements/kal_dp-original.txt
```

然后把环境解释器写入配置：

```json
{
  "python": {
    "kal_dp": "/home/USER/miniconda3/envs/kal_dp/bin/python"
  }
}
```

注意：固定版本的 PyTorch 未必适配接收方服务器的 CUDA。安装失败时应先根据服务器 CUDA/驱动安装合适的 PyTorch，再安装 requirements 中的其余依赖。
