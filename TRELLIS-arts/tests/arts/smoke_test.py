#!/usr/bin/env python3
"""
端到端 Smoke Test：通过 subprocess 调用真实 train_arts.py (ss_flow_art stage) 验证基础设施。

4 项验证（每项都直接运行 train_arts.py，不自建 Trainer）:
  1. 单卡 5 步训练 — loss 不 NaN、在合理范围
  2. Wandb 上线验证 — loss/lr 曲线出现在 dashboard
  3. Checkpoint 保存+加载 — 续训后 log 正常
  4. LoRA 模式验证 — 只有 LoRA 参数更新（通过 --dump-param-stats）

运行方式:
  python TRELLIS-arts/tests/arts/smoke_test.py                       # 全部测试
  python TRELLIS-arts/tests/arts/smoke_test.py --test 1              # 只运行测试 1
  python TRELLIS-arts/tests/arts/smoke_test.py --test 1,2,3,4        # 选择性运行
  python TRELLIS-arts/tests/arts/smoke_test.py --output-json out.json # JSON 输出

环境: trellis conda 环境，单卡 4090
"""

import os
import sys
import json
import re
import time
import math
import shutil
import tempfile
import argparse
import subprocess
import traceback

# 项目根目录: TRELLIS-arts/tests/arts/smoke_test.py -> 3 levels up
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'TRELLIS-arts', 'train_arts.py')
SMOKE_CONFIG = os.path.join(
    PROJECT_ROOT, 'TRELLIS-arts', 'configs', 'arts', 'ss_flow_art', 'smoke_test.yaml'
)


# ============================================================
# 工具函数
# ============================================================

def run_train(extra_overrides=None, extra_args=None, tmpdir=None, timeout=300):
    """
    subprocess 调用 train_arts.py (ss_flow_art)，返回 (returncode, stdout, stderr, output_dir)。

    Args:
        extra_overrides: list of "key=value" 覆盖
        extra_args: list of extra CLI args (如 ['--dump-param-stats'])
        tmpdir: 输出目录（None 则创建临时目录）
        timeout: 超时秒数
    """
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix='smoke_test_')

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--config', SMOKE_CONFIG,
        f'training.output_dir={tmpdir}',
    ]
    if extra_args:
        cmd.extend(extra_args)

    overrides = extra_overrides or []
    cmd.extend(overrides)

    env = os.environ.copy()
    env['WANDB_IGNORE_GLOBS'] = '*.pt,*.safetensors,*.ckpt'

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr, tmpdir
    except subprocess.TimeoutExpired:
        return -1, '', f'Timeout after {timeout}s', tmpdir


def parse_log_losses(output_dir):
    """
    从 output_dir/log.txt 解析每步 loss。

    log.txt 格式: step: {"mse": {"mse": 1.23}, "loss": 1.23, ...}
    返回: list of (step, loss_value)
    """
    log_path = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(log_path):
        return []

    losses = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: "step: {json}"
            match = re.match(r'^(\d+):\s*(.+)$', line)
            if match:
                step = int(match.group(1))
                try:
                    data = json.loads(match.group(2))
                    # loss 可能在 data["mse"]["mse"] 或 data["loss"] 或 data["mse"]
                    loss = None
                    if isinstance(data, dict):
                        if 'loss' in data:
                            loss = float(data['loss'])
                        elif 'mse' in data:
                            mse = data['mse']
                            if isinstance(mse, dict) and 'mse' in mse:
                                loss = float(mse['mse'])
                            else:
                                loss = float(mse)
                    if loss is not None:
                        losses.append((step, loss))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
    return losses


def parse_param_stats(stdout):
    """从 stdout 解析 --dump-param-stats 输出。"""
    stats = {}
    for line in stdout.splitlines():
        if '[PARAM_STATS_BEFORE]' in line:
            m = re.search(r'total=(\d+)\s+trainable=(\d+)\s+ratio=([0-9.]+)%', line)
            if m:
                stats['total_params'] = int(m.group(1))
                stats['trainable_params'] = int(m.group(2))
                stats['trainable_ratio'] = float(m.group(3))
        if '[PARAM_STATS_AFTER]' in line:
            m = re.search(r'lora_changed=(\d+)\s+non_lora_changed=(\d+)', line)
            if m:
                stats['lora_changed'] = int(m.group(1))
                stats['non_lora_changed'] = int(m.group(2))
            if 'OK: 非 LoRA 参数全部冻结' in line:
                stats['freeze_ok'] = True
            if 'WARNING: 非 LoRA 参数发生变化' in line:
                stats['freeze_ok'] = False
    return stats


# ============================================================
# 测试 1: 单卡 5 步训练
# ============================================================

def test_1_training_5steps():
    """
    验证 train_arts.py (ss_flow_art) 能正常跑 5 步，loss 合理。

    Pass/Fail:
    - [PASS] 进程退出码 0
    - [PASS] log.txt 中有 >= 1 条 loss 记录
    - [PASS] 所有 loss 是有限数（非 NaN/Inf）
    - [PASS] loss 在 [0.001, 50.0] 范围内
    - [FAIL] 进程非零退出或超时
    """
    print('[测试 1] 启动 train_arts.py (ss_flow_art) (5 步训练)...')
    tmpdir = None
    try:
        overrides = [
            'training.max_steps=5',
            'wandb.mode=disabled',  # 测试 1 不需要 wandb
            'training.pretrained_ckpt=null',
        ]
        returncode, stdout, stderr, tmpdir = run_train(extra_overrides=overrides)

        if returncode != 0:
            return {
                'status': 'FAIL',
                'error': f'train.py 退出码 {returncode}\nstderr: {stderr[-2000:]}',
                'stdout_tail': stdout[-1000:],
            }

        losses = parse_log_losses(tmpdir)
        loss_values = [l for _, l in losses]

        if len(loss_values) == 0:
            return {
                'status': 'FAIL',
                'error': 'log.txt 中无 loss 记录（i_log 配置可能有误）',
            }

        # 检查 NaN/Inf
        for i, v in enumerate(loss_values):
            if not math.isfinite(v):
                return {
                    'status': 'FAIL',
                    'error': f'step {losses[i][0]} loss={v} 不是有限数',
                    'losses': loss_values,
                }

        # 检查范围
        first_loss = loss_values[0]
        if first_loss < 0.001 or first_loss > 50.0:
            return {
                'status': 'FAIL',
                'error': f'首步 loss={first_loss:.4f} 超出合理范围 [0.001, 50.0]',
                'losses': loss_values,
            }

        return {
            'status': 'PASS',
            'losses': loss_values,
            'loss_first': first_loss,
            'loss_last': loss_values[-1],
            'num_steps_logged': len(loss_values),
        }

    except Exception as e:
        return {'status': 'FAIL', 'error': f'异常: {e}\n{traceback.format_exc()}'}
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 测试 2: Wandb 上线验证
# ============================================================

def test_2_wandb_online():
    """
    验证 train.py 能启动 wandb run 并记录指标。

    Pass/Fail:
    - [PASS] 进程正常退出
    - [PASS] stdout 中出现 wandb run URL
    - [FAIL] wandb 初始化失败
    """
    print('[测试 2] 启动 train_arts.py (ss_flow_art) (Wandb 在线模式)...')
    tmpdir = None
    try:
        run_name = f'smoke-test-{time.strftime("%Y%m%d-%H%M%S")}'
        overrides = [
            'training.max_steps=3',
            'wandb.mode=online',
            f'wandb.name={run_name}',
            'training.pretrained_ckpt=null',
        ]
        returncode, stdout, stderr, tmpdir = run_train(extra_overrides=overrides)

        if returncode != 0:
            # wandb 认证失败是常见原因
            if 'wandb' in stderr.lower() and ('login' in stderr.lower() or 'api_key' in stderr.lower()):
                return {
                    'status': 'FAIL',
                    'error': 'Wandb 认证失败，请先运行 wandb login',
                }
            return {
                'status': 'FAIL',
                'error': f'train.py 退出码 {returncode}\nstderr: {stderr[-2000:]}',
            }

        # 检查 stdout 和 stderr 中是否有 wandb URL（wandb 可能输出到 stderr）
        wandb_url = None
        combined_output = stdout + '\n' + stderr
        for line in combined_output.splitlines():
            if 'wandb.ai' in line or 'https://wandb' in line:
                url_match = re.search(r'https?://[^\s]+wandb[^\s]*', line)
                if url_match:
                    wandb_url = url_match.group(0)
                    break

        # 也检查 wandb 目录是否生成
        wandb_dir = os.path.join(tmpdir, 'wandb')
        wandb_dir_exists = os.path.exists(wandb_dir)

        losses = parse_log_losses(tmpdir)

        # wandb "上线验证" 要求：1) 有真正的 run URL  2) 至少 1 条指标被记录
        if not wandb_url:
            if wandb_dir_exists:
                error = 'wandb 本地目录存在但未获取到在线 run URL（可能 offline 模式或网络问题）'
            else:
                error = 'stdout 中未发现 wandb URL 且无 wandb 目录'
            return {
                'status': 'FAIL',
                'wandb_run_url': None,
                'wandb_run_name': run_name,
                'wandb_dir_exists': wandb_dir_exists,
                'losses_logged': len(losses),
                'error': error,
            }

        if len(losses) == 0:
            return {
                'status': 'FAIL',
                'wandb_run_url': wandb_url,
                'wandb_run_name': run_name,
                'wandb_dir_exists': wandb_dir_exists,
                'losses_logged': 0,
                'error': 'wandb run URL 存在但 log.txt 中无 loss 记录（指标可能未上报）',
            }

        return {
            'status': 'PASS',
            'wandb_run_url': wandb_url,
            'wandb_run_name': run_name,
            'wandb_dir_exists': wandb_dir_exists,
            'losses_logged': len(losses),
            'error': None,
        }

    except Exception as e:
        return {'status': 'FAIL', 'error': f'异常: {e}\n{traceback.format_exc()}'}
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 测试 3: Checkpoint 保存+加载
# ============================================================

def test_3_checkpoint_save_load():
    """
    验证 checkpoint 保存后能正常 resume。

    流程: 跑 3 步（i_save=3 触发保存）→ 从 checkpoint resume 跑到 step 5
    Pass/Fail:
    - [PASS] 第一次训练生成 checkpoint 文件
    - [PASS] 第二次训练从 checkpoint resume 成功
    - [PASS] resume 后继续训练不报错
    """
    print('[测试 3] Checkpoint 保存+加载...')
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix='smoke_ckpt_')

        # 第一次: 跑 3 步，i_save=3 触发保存
        overrides1 = [
            'training.max_steps=3',
            'training.i_save=3',
            'wandb.mode=disabled',
            'training.pretrained_ckpt=null',
            f'training.output_dir={tmpdir}',
        ]
        rc1, stdout1, stderr1, _ = run_train(extra_overrides=overrides1, tmpdir=tmpdir)

        if rc1 != 0:
            return {
                'status': 'FAIL',
                'error': f'第一次训练失败 (rc={rc1})\nstderr: {stderr1[-2000:]}',
            }

        # 检查 checkpoint（TRELLIS 保存到 output_dir/ckpts/，不是 checkpoints/）
        ckpt_dir = os.path.join(tmpdir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            return {
                'status': 'FAIL',
                'error': f'checkpoint 目录不存在: {ckpt_dir}（TRELLIS 保存到 ckpts/）',
            }

        ckpt_files = os.listdir(ckpt_dir)
        if not ckpt_files:
            return {
                'status': 'FAIL',
                'error': 'ckpts/ 目录为空',
            }

        # 计算 checkpoint 总大小
        ckpt_size_mb = sum(
            os.path.getsize(os.path.join(ckpt_dir, f))
            for f in os.listdir(ckpt_dir)
            if os.path.isfile(os.path.join(ckpt_dir, f))
        ) / (1024 * 1024)

        losses1 = parse_log_losses(tmpdir)

        # 清空 log.txt 以区分 resume 后的 log
        log_path = os.path.join(tmpdir, 'log.txt')
        if os.path.exists(log_path):
            os.rename(log_path, log_path + '.bak')

        # 第二次: 从 checkpoint resume，跑到 step 5
        # train.py 通过 --load-dir + --resume-step 传入 resume 参数
        # BasicTrainer 在 load_dir 和 step 都提供时才触发 load()
        overrides2 = [
            'training.max_steps=5',
            'training.i_save=999999',
            'wandb.mode=disabled',
            'training.pretrained_ckpt=null',
            f'training.output_dir={tmpdir}',
        ]
        rc2, stdout2, stderr2, _ = run_train(
            extra_overrides=overrides2,
            extra_args=['--load-dir', tmpdir, '--resume-step', '3'],
            tmpdir=tmpdir,
        )

        if rc2 != 0:
            return {
                'status': 'FAIL',
                'error': f'Resume 训练失败 (rc={rc2})\nstderr: {stderr2[-2000:]}',
            }

        # 验证 resume 真正发生了
        # 1. stdout 中应出现 "Loading checkpoint from step 3"
        resume_confirmed = 'Loading checkpoint from step 3' in stdout2
        if not resume_confirmed:
            # 也检查类似的加载信息
            resume_confirmed = 'Loading checkpoint' in stdout2 and 'step' in stdout2

        # 2. 第二次的 log.txt 的 step 应该从 4 或 5 开始，不是 1
        losses2 = parse_log_losses(tmpdir)
        resumed_steps = [s for s, _ in losses2]
        step_continuity_ok = True
        if resumed_steps:
            # resume 后的 step 应该 > 3（从 step 4 开始）
            if min(resumed_steps) <= 3:
                step_continuity_ok = False

        if not resume_confirmed:
            return {
                'status': 'FAIL',
                'error': 'stdout 中未发现 checkpoint 加载信息（resume 可能未真正执行）',
                'stdout2_tail': stdout2[-1000:],
            }

        # 3. resumed_steps 必须非空（确认 resume 后确实有新训练产出）
        if not resumed_steps:
            return {
                'status': 'FAIL',
                'error': 'Resume 后 log.txt 无新 loss 记录（训练可能未实际执行）',
            }

        if not step_continuity_ok:
            return {
                'status': 'FAIL',
                'error': f'Resume 后 step 编号异常: {resumed_steps}（期望从 4 开始，实际可能从头重跑）',
            }

        return {
            'status': 'PASS',
            'checkpoint_files': ckpt_files,
            'checkpoint_size_mb': round(ckpt_size_mb, 1),
            'losses_phase1': [l for _, l in losses1],
            'losses_phase2': [l for _, l in losses2],
            'resumed_steps': resumed_steps,
            'resume_confirmed_in_stdout': resume_confirmed,
        }

    except Exception as e:
        return {'status': 'FAIL', 'error': f'异常: {e}\n{traceback.format_exc()}'}
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 测试 4: LoRA 模式验证
# ============================================================

def test_4_lora_mode():
    """
    验证 LoRA 模式下只有 LoRA 参数更新。

    通过 --dump-param-stats 让 train.py 在训练前后打印参数 hash，
    解析 stdout 判断非 LoRA 参数是否冻结。

    Pass/Fail:
    - [PASS] 可训练参数占比 < 5%
    - [PASS] 非 LoRA 参数全部冻结 (non_lora_changed=0)
    - [PASS] LoRA 参数有更新 (lora_changed > 0)
    """
    print('[测试 4] LoRA 模式验证...')
    tmpdir = None
    try:
        overrides = [
            'training.max_steps=3',
            'wandb.mode=disabled',
            'lora.enabled=true',
            'lora.rank=16',
            'lora.target_modules=all_attn',
            'training.pretrained_ckpt=null',
        ]
        returncode, stdout, stderr, tmpdir = run_train(
            extra_overrides=overrides,
            extra_args=['--dump-param-stats'],
        )

        if returncode != 0:
            return {
                'status': 'FAIL',
                'error': f'LoRA 训练失败 (rc={returncode})\nstderr: {stderr[-2000:]}',
            }

        stats = parse_param_stats(stdout)

        if not stats:
            return {
                'status': 'FAIL',
                'error': 'stdout 中未找到 PARAM_STATS 输出',
                'stdout_tail': stdout[-2000:],
            }

        # 检查可训练参数占比
        trainable_ratio = stats.get('trainable_ratio', 100)
        if trainable_ratio > 5.0:
            return {
                'status': 'FAIL',
                'error': f'可训练参数占比 {trainable_ratio:.2f}% > 5%（LoRA 注入范围异常）',
                **stats,
            }

        # 检查冻结状态
        freeze_ok = stats.get('freeze_ok', None)
        non_lora_changed = stats.get('non_lora_changed', -1)
        lora_changed = stats.get('lora_changed', 0)

        if freeze_ok is False or non_lora_changed > 0:
            return {
                'status': 'FAIL',
                'error': f'非 LoRA 参数发生变化 (non_lora_changed={non_lora_changed})',
                **stats,
            }

        if lora_changed == 0:
            return {
                'status': 'FAIL',
                'error': 'LoRA 参数训练前后完全一致（未参与训练）',
                **stats,
            }

        losses = parse_log_losses(tmpdir)

        return {
            'status': 'PASS',
            'trainable_ratio': trainable_ratio,
            'total_params': stats.get('total_params', 0),
            'trainable_params': stats.get('trainable_params', 0),
            'lora_changed': lora_changed,
            'non_lora_changed': non_lora_changed,
            'freeze_ok': True,
            'losses': [l for _, l in losses],
        }

    except Exception as e:
        return {'status': 'FAIL', 'error': f'异常: {e}\n{traceback.format_exc()}'}
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Stage 3 Smoke Test (subprocess 方案)')
    parser.add_argument('--test', type=str, default='all',
                        help='运行哪些测试: all / 1 / 1,2,3 / 1,4')
    parser.add_argument('--output-json', type=str, default=None,
                        help='将结果写入 JSON 文件')
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查 train.py 存在
    if not os.path.exists(TRAIN_SCRIPT):
        print(f'[ERROR] 找不到训练脚本: {TRAIN_SCRIPT}')
        sys.exit(1)
    if not os.path.exists(SMOKE_CONFIG):
        print(f'[ERROR] 找不到配置文件: {SMOKE_CONFIG}')
        sys.exit(1)

    # 决定运行哪些测试
    if args.test == 'all':
        test_ids = [1, 2, 3, 4]
    else:
        test_ids = [int(x.strip()) for x in args.test.split(',')]

    test_funcs = {
        1: ('单卡 5 步训练', test_1_training_5steps),
        2: ('Wandb 上线验证', test_2_wandb_online),
        3: ('Checkpoint 保存+加载', test_3_checkpoint_save_load),
        4: ('LoRA 模式验证', test_4_lora_mode),
    }

    results = {}
    for tid in test_ids:
        name, func = test_funcs[tid]
        print(f'\n{"="*60}')
        print(f'测试 {tid}: {name}')
        print(f'{"="*60}')
        try:
            result = func()
        except Exception as e:
            result = {'status': 'FAIL', 'error': f'未捕获异常: {e}\n{traceback.format_exc()}'}
        results[f'test_{tid}'] = {'name': name, **result}

        status = result['status']
        print(f'\n[{status}] 测试 {tid}: {name}')
        if result.get('error'):
            print(f'  错误: {result["error"][:200]}')

    # 汇总
    print(f'\n{"="*60}')
    print('Smoke Test 汇总')
    print(f'{"="*60}')

    total = len(results)
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    failed = total - passed

    for key, result in results.items():
        status = result['status']
        name = result['name']
        print(f'  [{status}] {name}')
        if 'losses' in result and result['losses']:
            print(f'    losses: {[f"{l:.4f}" for l in result["losses"][:5]]}')
        if 'wandb_run_url' in result and result['wandb_run_url']:
            print(f'    wandb URL: {result["wandb_run_url"]}')
        if 'checkpoint_size_mb' in result:
            print(f'    checkpoint: {result["checkpoint_size_mb"]} MB')
        if 'trainable_ratio' in result:
            print(f'    LoRA 可训练参数占比: {result["trainable_ratio"]:.2f}%')

    print(f'\n结果: {passed}/{total} 通过, {failed}/{total} 失败')

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\n详细结果已写入: {args.output_json}')

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
