#!/usr/bin/env bash

LIST_FILE="./down.txt"
DEST_DIR="./down_13B_GSVA"
PREFIX=""

# 如果清单里有相对路径，就给它补这个公共前缀目录
SRC_ROOT="/gemini/space/zhaozy/libingyu/GSVA-main/vis_GSVA_output_batch_13B_val"

mkdir -p "$DEST_DIR"

while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行
    [ -z "$line" ] && continue

    # 绝对路径直接用；相对路径补上 SRC_ROOT
    if [[ "$line" = /* ]]; then
        src="$line"
    else
        src="$SRC_ROOT/$line"
    fi

    if [ ! -e "$src" ]; then
        echo "跳过，不存在: $src"
        continue
    fi

    name="$(basename "$src")"
    cp -r "$src" "$DEST_DIR/${PREFIX}${name}"
    echo "已复制: $src -> $DEST_DIR/${PREFIX}${name}"
done < "$LIST_FILE"