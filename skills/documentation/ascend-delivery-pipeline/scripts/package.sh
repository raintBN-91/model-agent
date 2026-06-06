#!/bin/bash
# ascend-delivery-pipeline: standalone package script
# Usage: bash scripts/package.sh [target_dir]

set -e

TARGET_DIR="${1:-$(pwd)}"
cd "$TARGET_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DELIVERY_DIR="deliverables_${TIMESTAMP}"
mkdir -p "$DELIVERY_DIR"

echo "=== ascend-delivery-pipeline: packaging deliverables ==="
echo "Source: $TARGET_DIR"
echo "Dest:   $DELIVERY_DIR"

# Copy deliverable categories
copy_safe() {
    for f in "$@"; do
        if [ -f "$f" ]; then
            cp -v "$f" "$DELIVERY_DIR/" 2>/dev/null || true
        fi
    done
}

copy_safe README* readme* *.py *.log *.json *.md *.yaml *.yml

# Generate MANIFEST
cat > "$DELIVERY_DIR/MANIFEST.md" <<EOF
# 交付清单

| 类别 | 文件 | 说明 |
|------|------|------|
EOF

for f in $(ls -1 "$DELIVERY_DIR/"); do
    case "$f" in
        README*|readme*) DESC="模型部署与适配说明文档" ;;
        *.py) DESC="推理/验证脚本" ;;
        *.log) DESC="运行日志" ;;
        *.json) DESC="结构化数据/报告" ;;
        *.md) DESC="补充文档" ;;
        *.yaml|*.yml) DESC="配置文件" ;;
        *) DESC="其他" ;;
    esac
    BASENAME=$(echo "$f" | sed 's/\.[^.]*$//')
    echo "| $BASENAME | $f | $DESC |" >> "$DELIVERY_DIR/MANIFEST.md"
done

echo ""
echo "Package created: $DELIVERY_DIR/"
echo "Files included:"
ls -1 "$DELIVERY_DIR/"
