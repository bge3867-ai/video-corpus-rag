"""从视频库彻底删除一个视频: 视频文件 + 切片 + Chroma 向量 (主库和文本库)。

用法: python tools/cleanup_video.py upload_test.mp4
(传文件名即可, 自动到 videos 目录找)
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import chromadb

from common import load_cfg
from index import md5_of_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="视频文件名 (videos 目录下) 或绝对路径")
    args = ap.parse_args()

    cfg = load_cfg()
    vpath = (
        args.video
        if os.path.isabs(args.video)
        else os.path.join(cfg["paths"]["videos"], os.path.basename(args.video))
    )
    if not os.path.isfile(vpath):
        print("文件不存在:", vpath)
        return 1

    vmd5 = md5_of_file(vpath)
    idx = cfg["index"]
    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_db"])
    n_del = 0
    for cname in (idx["collection"], idx["text_collection"]):
        col = client.get_or_create_collection(cname)
        rows = col.get(where={"video_md5": vmd5}, include=[])
        if rows["ids"]:
            col.delete(ids=rows["ids"])
            n_del += len(rows["ids"])
            print(f"{cname}: 删除 {len(rows['ids'])} 条向量")

    stem = os.path.splitext(os.path.basename(vpath))[0]
    cdir = os.path.join(cfg["paths"]["clips"], stem)
    if os.path.isdir(cdir):
        shutil.rmtree(cdir)
        print("已删片段目录:", cdir)

    os.remove(vpath)
    print(f"已删视频文件: {vpath} (共清理 {n_del} 条向量)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
