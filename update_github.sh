#!/bin/bash
# ============================================================
# 一键更新 GitHub: 提交当前所有改动并推送
# 用法:
#   bash update_github.sh                # 使用默认提交说明
#   bash update_github.sh "修复了xxx"     # 自定义提交说明
# 前提: 本机已用 gh 登录 GitHub (gh auth status 查看)
# ============================================================
cd "$(dirname "$0")" || exit 1
MSG="${1:-更新代码}"

git add -A
if git diff --cached --quiet; then
  echo "没有需要提交的改动"
else
  git commit -m "$MSG"
fi
git push
echo "== 已推送到 GitHub: https://github.com/bge3867-ai/video-corpus-rag =="
