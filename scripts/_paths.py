"""脚本引导：把项目的 src/ 加入 sys.path，使 `import esp_xform` 可用。

每个脚本在最顶部 `import _paths  # noqa` 即可，无需安装包。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROJECT_ROOT = _PROJECT_ROOT
SRC_DIR = _SRC
