"""Machine learning library tailored for data streams."""

from ._prepare_jpype import _start_jpype, about
from .__about__ import __version__

# It is important that this is called before importing any other module
_start_jpype()
# **_start_jpype()**​：调用函数启动JPype（Java虚拟机环境）。
# 必须在导入其他模块之前调用，确保Java环境准备好。这表明库的底层功能依赖Java（如MOA/JDM等数据流处理框架），需通过JPype桥接。


from . import stream  # noqa Module imported here to ensure that jpype has been started
#  ​**noqa**​：忽略代码检查工具的警告（如"导入未在顶部"的提示），因为_start_jpype()必须优先执行。
#  此时导入stream模块，是因为它依赖于已初始化的JPype环境。

__all__ = [
    "about",
    "__version__",
    "stream",
]
