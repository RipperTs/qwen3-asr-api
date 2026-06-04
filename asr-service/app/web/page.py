import os

_DIR = os.path.dirname(__file__)

# 读取 HTML 模板文件
HTML_PAGE = open(os.path.join(_DIR, "index.html"), encoding="utf-8").read()

# 实时转写测试页
STREAM_PAGE = open(os.path.join(_DIR, "stream.html"), encoding="utf-8").read()
