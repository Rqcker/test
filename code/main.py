# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/6/10
import subprocess


def main():
    subprocess.call(
        "python uag.py \
            --task GSM8K \
            --data-path GSM8K_input.jsonl \
            --record-path GSM8K_output.jsonl \
            --demonstration-path GSM8K_demonstration.jsonl",
        shell=True,
    )


if __name__ == "__main__":
    main()
