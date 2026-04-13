import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

def find_first_specific_so_file(root_dir,file_name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件名是否以 'arx_lift_python' 开头，并且以 '.so' 结尾
            if filename.startswith(file_name) and filename.endswith('.so'):
                # 返回第一个找到的符合条件的文件路径
                return os.path.join(dirpath, filename)
    return None  # 如果没有找到符合条件的文件，则返回 None

# 使用 os.path.join 来拼接路径
so_file = find_first_specific_so_file(os.path.join(current_dir, 'api', 'arx_lift_python'),'arx_lift_python.')

# 确保共享库的路径在 Python 的路径中
if os.path.exists(so_file):
    sys.path.append(os.path.dirname(so_file))  # 添加共享库所在目录到 sys.path
else:
    raise FileNotFoundError(f"Shared library not found: {so_file}")