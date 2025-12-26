import json
import os.path

import jwt
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

from src.engine_utils.directory_info import DirectoryInfo

router = APIRouter()
security = HTTPBearer()

# JWT配置
JWT_SECRET = "human@123"
JWT_ALGORITHM = "HS256"


class LoginRequest(BaseModel):
    """ 登录请求参数 """
    account: str
    secret: str


def load_users():
    """从user.json加载用户数据"""
    try:
        user_file = os.path.join(DirectoryInfo.get_config_dir(), "user.json")
        with open(user_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def get_user(account: str):
    users = load_users()
    for user in users:
        if user["account"] == account:
            return user


def verify_password(plain_password, hashed_password):
    """验证密码（这里假设密码已哈希存储）"""
    return plain_password == hashed_password


def create_access_token(data: dict, expires_at: Optional[datetime] = None):
    """创建JWT访问令牌"""
    to_encode = data.copy()
    if expires_at:
        expire = expires_at
    else:
        # 默认30天过期
        expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    """验证JWT令牌"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        account: str = payload.get("sub")
        if account is None:
            raise HTTPException(status_code=401, detail="Token无效")
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token无效")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户（依赖于JWT认证）"""
    token = credentials.credentials
    return verify_token(token)


@router.post("/login")
async def login(request: LoginRequest):
    """用户登录接口"""
    # 查找用户
    user = get_user(request.account)

    if not user or not verify_password(request.secret, user["secret"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 检查是否过期
    expire_time = datetime.strptime(user["expireTime"], "%Y-%m-%d %H:%M:%S")
    if datetime.now() > expire_time:
        raise HTTPException(status_code=401, detail="账户已过期")

    # 创建访问令牌
    access_token = create_access_token(
        data={"sub": user["account"], "organization": user["organization"]},
        expires_at=expire_time
    )

    return {
        "access_token": access_token,
        "token_type": "Bearer ",
        "account": user["account"],
        "organization": user["organization"]
    }


@router.get("/verify_token")
async def verify_token_endpoint(current_user: dict = Depends(get_current_user)):
    """验证token接口"""
    return {
        "account": current_user.get("sub"),
        "organization": current_user.get("organization")
    }
