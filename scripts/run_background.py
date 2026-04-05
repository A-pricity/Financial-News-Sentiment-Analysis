#!/usr/bin/env python3
"""
后台运行脚本 - 支持训练和评估任务的后台运行
用法:
    python run_background.py --mode train          # 后台训练
    python run_background.py --mode eval          # 后台评估
    python run_background.py --status              # 查看运行状态
    python run_background.py --stop                # 停止运行中的任务
"""

import os
import sys
import signal
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PID_FILE = os.path.join(PROJECT_ROOT, ".running.pid")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def get_running_pid():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    return None


def is_process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_status():
    pid = get_running_pid()
    if pid is None:
        print("没有正在运行的任务")
        return

    if is_process_running(pid):
        print(f"任务正在运行 (PID: {pid})")

        log_files = sorted(
            Path(LOG_DIR).glob("train_*.log"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if log_files:
            latest_log = log_files[0]
            print(f"日志文件: {latest_log}")
            print("\n最近日志 (最后10行):")
            with open(latest_log, "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip())
    else:
        print(f"PID文件存在但进程 {pid} 已停止")
        os.remove(PID_FILE)


def stop_task():
    pid = get_running_pid()
    if pid is None:
        print("没有正在运行的任务")
        return

    if is_process_running(pid):
        print(f"正在停止进程 {pid}...")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            if is_process_running(pid):
                os.kill(pid, signal.SIGKILL)
            print("进程已停止")
        except ProcessLookupError:
            print("进程已停止")
    else:
        print(f"进程 {pid} 不在运行")

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def start_task(mode, args_list=None):
    if args_list is None:
        args_list = []

    pid = get_running_pid()
    if pid and is_process_running(pid):
        print(f"已有任务正在运行 (PID: {pid})")
        print("请先停止当前任务: python run_background.py --stop")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "train":
        script = os.path.join(SCRIPT_DIR, "train.py")
        out_file = os.path.join(LOG_DIR, f"train_{timestamp}.out")
    elif mode == "eval":
        script = os.path.join(SCRIPT_DIR, "evaluate.py")
        out_file = os.path.join(LOG_DIR, f"eval_{timestamp}.out")
    else:
        print(f"未知模式: {mode}")
        return

    cmd = ["uv", "run", "python", script] + args_list

    with open(out_file, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )

    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    print(f"任务已启动 (PID: {proc.pid})")
    print(f"输出文件: {out_file}")
    print(f"查看日志: tail -f {out_file}")
    print(f"或查看解析日志: tail -f logs/train_*.log")


def main():
    parser = argparse.ArgumentParser(description="后台运行脚本")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], help="运行模式")
    parser.add_argument("--stop", action="store_true", help="停止当前任务")
    parser.add_argument("--status", action="store_true", help="查看运行状态")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--resume", type=str, help="从检查点恢复")
    parser.add_argument("--cv", type=int, default=0, help="K折交叉验证")
    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.stop:
        stop_task()
    elif args.mode:
        extra_args = []
        if args.epochs:
            extra_args.extend(["--epochs", str(args.epochs)])
        if args.resume:
            extra_args.extend(["--resume", args.resume])
        if args.cv > 0:
            extra_args.extend(["--cv", str(args.cv)])
        start_task(args.mode, extra_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
