#coding=utf-8
from distutils.core import setup
import py2exe
includes = ["encodings", "encodings.*"]
#要包含的其它库文件
options = {"py2exe":
  {
    "compressed": 1, #压缩
    "optimize": 2,
    "includes": includes,
    "bundle_files": 1 #所有文件打包成一个exe文件
  }
}
setup (
  options = options,
  zipfile=None,  #不生成library.zip文件
  console=[{"script": "01program.py", "icon_resources": [(1, "01.jpg")] }]#源文件，程序图标
  #console = [{"script": "01program.py"}]# 源文件
)
