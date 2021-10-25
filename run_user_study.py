"""Script for easily running user-study interfaces"""
import os
import subprocess
import time
from utils import mkdir_fresh
from pathlib import Path

def run_cmd(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def main():
    # This script is helpful in quickly generating affordance map page
    print("Please enter the save folder name for user:")
    debug_path_name = input()
    debug_root = Path(f"real_world/debug/{debug_path_name}")
    mkdir_fresh(debug_root, ask_first=False)
    while True:
        try:
            run_cmd("pkill -9 -f 'python real_world/baseline.py'")
        except Exception as e:
            pass
        try:
            run_cmd("pkill -9 -f 'python real_world/main.py'")
        except Exception as e:
            pass

        print("==============================================")
        print("==============================================")
        print("==============================================")
        print("==============================================")
        print("==============================================")
        while True:
            print("Press 1 for running TeleSnap; 2 for running Baseline.")
            interface = input()
            if interface not in ["1", "2"]:
                print("invalid interface")
            else:
                break
        script = "main.py" if interface == "1" else "baseline.py"
        cmd = f"python real_world/{script} perception.debug_path_name={debug_path_name}"
        run_cmd(cmd)
        print("command returned")


if __name__ == "__main__":
    main()
