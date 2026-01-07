import json
import os
import threading
from datetime import datetime
from src.engine_utils.directory_info import DirectoryInfo


class TokenManager:
    def __init__(self):
        self.active_tokens = {}
        self.lock = threading.Lock()
        self.token_file = os.path.join(DirectoryInfo.get_config_dir(), "active_tokens.json")
        self.load_tokens()

    def load_tokens(self):
        """从文件加载活跃token"""
        try:
            with open(self.token_file, "r", encoding="utf-8") as f:
                self.active_tokens = json.load(f)
        except FileNotFoundError:
            self.active_tokens = {}

    def save_tokens(self):
        """将活跃token保存到文件"""
        with self.lock:
            with open(self.token_file, "w", encoding="utf-8") as f:
                json.dump(self.active_tokens, f, ensure_ascii=False, indent=2)

    def store_user_token(self, account: str, token: str, expire_time: datetime):
        """存储用户token，实现互斥登录"""
        # 删除旧token记录
        if account in self.active_tokens:
            old_token = self.active_tokens[account]["token"]
            # 从反向映射中删除旧token
            if old_token in [v["token"] for v in self.active_tokens.values()]:
                pass  # 保持一致性，实际存储在字典键值中

        # 存储新token
        self.active_tokens[account] = {
            "token": token,
            "expire_time": expire_time.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        self.save_tokens()

    def is_token_valid(self, token: str) -> bool:
        """检查token是否仍然有效"""
        for account, token_info in self.active_tokens.items():
            if token_info["token"] == token:
                # 检查是否过期
                expire_time = datetime.fromisoformat(token_info["expire_time"])
                if datetime.now() > expire_time:
                    # 从存储中删除过期token
                    self.remove_expired_token(account)
                    return False
                return True
        return False

    def remove_expired_token(self, account: str):
        """移除过期的token"""
        if account in self.active_tokens:
            del self.active_tokens[account]
            self.save_tokens()
