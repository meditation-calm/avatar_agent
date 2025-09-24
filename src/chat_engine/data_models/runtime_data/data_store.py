from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class DataStoreType(IntEnum):
    """
    数据存储类型的枚举：
        INVALID = 0：无效存储类型，表示数据存储未初始化或无效
        LOCAL_MEMORY = 1：本地内存存储类型，表示数据存储在本地内存中
    """
    INVALID = 0
    LOCAL_MEMORY = 1


@dataclass
class DataStore:
    """
    数据存储容器，用于封装实际的数据和其存储信息：
        data：实际存储的数据，类型为 Any，可以是任何Python对象
        storage：存储类型，使用 DataStoreType 枚举
    """
    data: Any
    storage: DataStoreType = DataStoreType.INVALID

    @property
    def valid(self):
        return self.storage != DataStoreType.INVALID

    def set_data(self, data: Any, storage: DataStoreType):
        """设置数据和存储类型"""
        self.data = data
        self.storage = storage

    def get_data(self):
        """获取存储的数据"""
        return self.data
