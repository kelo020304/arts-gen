"""
Wandb 日志 Mixin：在 MRO 中放在 BasicTrainer 之前，拦截日志和快照事件。

用法:
    class MyTrainer(WandbMixin, SomeBaseTrainer):
        pass

    trainer = MyTrainer(models=..., dataset=..., wandb_config={...}, ...)
"""

import os
from typing import Optional


class WandbMixin:
    """Wandb 日志 Mixin。

    通过 kwargs 接收 wandb_config，在 MRO 中放在 BasicTrainer 之前。
    如果 wandb_config 为 None、未传入，或未显式设置 enabled=true，
    则所有 wandb 操作静默跳过。

    wandb_config 示例:
        {
            "project": "arts-reconstruction",
            "name": "stage3-mv4-exp01",
            "tags": ["stage3", "mv4"],
            "enabled": true,        # default false
            "mode": "online",       # "online" / "offline" / "disabled"
        }
    """

    def __init__(self, *args, wandb_config: Optional[dict] = None, **kwargs):
        # 先 pop 出 wandb_config，不传给父类
        self._wandb_config = wandb_config
        self._wandb_run = None
        super().__init__(*args, **kwargs)
        # 在父类 __init__ 完成后（rank 已知）初始化 wandb
        self._init_wandb()

    def _init_wandb(self):
        """仅在 rank 0 时初始化 wandb run。"""
        if self._wandb_config is None:
            return
        if not bool(self._wandb_config.get('enabled', False)):
            return
        if not getattr(self, 'is_master', True):
            return

        try:
            import wandb
        except ImportError:
            print('[WARN] wandb 未安装，跳过 wandb 初始化')
            return

        mode = self._wandb_config.get('mode', 'disabled')
        if mode == 'disabled':
            return
        self._wandb_run = wandb.init(
            project=self._wandb_config.get('project', 'arts-reconstruction'),
            name=self._wandb_config.get('name', None),
            tags=self._wandb_config.get('tags', None),
            config=self._wandb_config.get('config', None),
            dir=getattr(self, 'output_dir', None),
            mode=mode,
            resume='allow',
        )

    def _on_log(self, metrics: dict, step: int):
        """记录标量指标到 wandb。"""
        if self._wandb_run is None:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f'[WARN] wandb log 失败: {e}')

    def _on_snapshot(self, sample_dir: str, step: int):
        """将快照图片上传到 wandb。"""
        if self._wandb_run is None:
            return
        try:
            import wandb
            images = {}
            for fname in os.listdir(sample_dir):
                if fname.endswith(('.jpg', '.png')):
                    images[fname] = wandb.Image(os.path.join(sample_dir, fname))
            if images:
                wandb.log(images, step=step)
        except Exception as e:
            print(f'[WARN] wandb snapshot 上传失败: {e}')
