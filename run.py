#!/usr/bin/env python3
"""本地开发启动脚本。"""

import sys
import uvicorn

if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=18003, reload=True)
