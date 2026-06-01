## 1. 虚拟环境
- 本项目的 Python 运行环境为：`D:\anaconda3\envs\article`
- 在执行任何 Python 脚本或安装依赖时，请默认使用该环境的解释器，例如：
  ```bash
  D:\anaconda3\envs\article\python.exe script.py
## 2. 语言与称呼
回答问题时，必须使用中文，只有完全不可控的部分（如第三方库的英文输出、代码报错原文等）可以保留英文，以便我检查过程和理解问题。

在所有对话中，请用“小红桃”来称呼我。

## 3. GitHub 仓库与推送规则
项目远程仓库地址：https://github.com/hangtaox/Bathymetry-Inversion-MLP-UNet-.git

代码推送原则：只推送纯代码文件，不推送数据、图片、训练权重等大文件，避免占用过大空间。

需要排除的内容包括但不限于：

数据文件：.nc、.csv、.tif、.mat、.npy、.npz 等

图片文件：.png、.jpg、.jpeg、.tiff、.bmp 等

模型权重：.pth、.ckpt、.h5、.pb、.onnx、.pt 等

其他大文件或自动生成的文件夹：__pycache__/、.ipynb_checkpoints/、logs/、outputs/ 等

在执行 git push 前，请务必检查暂存区，确保只包含代码文件（如 .py、.yaml、.md、.txt、.gitignore 等），并建议维护合适的 .gitignore 文件。

如果我要求“推送到仓库”或“备份到 GitHub”，请先列出即将推送的文件清单并征得我确认，然后再推送。