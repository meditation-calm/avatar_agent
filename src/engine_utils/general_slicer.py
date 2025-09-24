from dataclasses import dataclass
from typing import Callable, Any, List

import numpy as np
from loguru import logger


@dataclass
class SliceManipulator:
    """
    定义数据切片操作的通用接口，支持对不同类型的数据进行切片操作。
        size_func: 计算数据大小的函数
        slice_func: 执行切片操作的函数
        concat_func: 连接多个数据片段的函数
    """
    size_func: Callable[[Any], int]
    slice_func: Callable[[Any, int, int], Any]
    concat_func: Callable[[List[Any]], Any]

    @classmethod
    def create_numpy_manipulator(cls, axis: int):
        """
        创建针对 NumPy 数组的切片操作器
            定义内部函数 slice_numpy 用于沿指定轴进行切片操作
            创建 SliceManipulator 实例，设置针对 NumPy 数组的操作函数：
            size_func: 获取指定轴的维度大小
            slice_func: 沿指定轴切片
            concat_func: 沿指定轴连接多个数组
        """
        def slice_numpy(x, start, end, slice_axis):
            indices = [slice(None)] * len(x.shape)
            indices[slice_axis] = slice(start, end)
            return x[tuple(indices)]

        manipulator = SliceManipulator(
            size_func=lambda x: x.shape[axis],
            slice_func=lambda x, start, end: slice_numpy(x, start, end, axis),
            concat_func=lambda x: np.concatenate(x, axis=axis),
        )
        return manipulator


@dataclass
class SliceContext:
    """
    维护切片操作的上下文状态，跟踪切片进度和剩余数据。
        slice_size: 每个切片的大小
        data_manipulator: 数据操作器
        last_remainder: 上次切片后剩余的数据
        sliced_sample_num: 已处理的数据样本数
        next_slice_start_id: 下一个切片的起始 ID
        last_slice_size: 上一个切片的大小
    """
    slice_size: int
    data_manipulator: SliceManipulator
    last_remainder: Any = None
    sliced_sample_num: int = 0  # num of input data
    next_slice_start_id: int = 0
    last_slice_size: int = 0

    @classmethod
    def create_numpy_slice_context(cls, slice_size: int, slice_axis: int):
        """
        创建针对 NumPy 数组的切片上下文
            创建 SliceContext 实例
            使用 create_numpy_manipulator 创建数据操作器
        """
        context = SliceContext(
            slice_size=slice_size,
            data_manipulator=SliceManipulator.create_numpy_manipulator(slice_axis),
        )
        return context

    def flush(self):
        """
        清空上下文状态并返回剩余数据
            保存当前剩余数据
            重置所有状态变量
            返回剩余数据
        """
        remainder = self.last_remainder
        self.last_remainder = None
        self.sliced_sample_num = 0
        self.next_slice_start_id = 0
        self.last_slice_size = 0
        return remainder

    def update_start_id(self, data_start_id: int, force_update: bool = False):
        """
        更新切片起始 ID
            当尚未处理任何数据或强制更新时，设置新的起始 ID
            记录日志信息
        """
        if self.sliced_sample_num == 0 or force_update:
            self.next_slice_start_id = data_start_id
            logger.warning(f"Update slicer start id to {data_start_id}")

    def get_last_slice_start_index(self):
        """
        获取上一个切片的起始索引
            通过当前起始 ID 减去上一个切片大小计算得出
        """
        return self.next_slice_start_id - self.last_slice_size

    def get_next_slice_start_index(self):
        """ 获取下一个切片的起始索引 """
        return self.next_slice_start_id


def slice_data(context: SliceContext, data):
    """ 对输入数据进行切片处理，生成固定大小的数据片段
            处理上次切片的剩余数据
            计算输入数据大小并更新已处理样本数
            循环生成固定大小的切片：
                处理跨剩余数据和当前数据的切片情况
                使用数据操作器执行实际切片操作
                更新切片索引和大小信息
                通过 yield 返回每个切片
                处理切片后剩余的数据，保存到上下文中供下次使用
    """
    # TODO update slice start id
    slice_func = context.data_manipulator.slice_func

    remainder_size = 0
    remainder_data = context.last_remainder
    context.last_remainder = None
    if remainder_data is not None:
        remainder_size = context.data_manipulator.size_func(remainder_data)
    input_size = context.data_manipulator.size_func(data)
    context.sliced_sample_num += input_size
    slice_start = -remainder_size
    while slice_start + context.slice_size <= input_size:
        outputs = []
        slice_end = slice_start + context.slice_size
        if slice_start < 0:
            if slice_end < 0:
                outputs.append(slice_func(remainder_data, remainder_size + slice_start, remainder_size + slice_end))
            else:
                outputs.append(slice_func(remainder_data, remainder_size + slice_start, remainder_size))
                if slice_end > 0:
                    outputs.append(slice_func(data, 0, slice_end))
        else:
            outputs.append(slice_func(data, slice_start, slice_end))

        if len(outputs) > 1:
            result = context.data_manipulator.concat_func(outputs)
        else:
            result = outputs[0]
        context.last_slice_size = context.data_manipulator.size_func(result)
        context.next_slice_start_id += context.last_slice_size
        yield result
        slice_start += context.slice_size
    remainders = []
    if slice_start < 0:
        remainders.append(slice_func(remainder_data, remainder_size + slice_start, remainder_size))
        if input_size > 0:
            remainders.append(data)
    elif slice_start < input_size:
        remainders.append(slice_func(data, slice_start, input_size))
    if len(remainders) > 1:
        context.last_remainder = context.data_manipulator.concat_func(remainders)
    elif len(remainders) == 1:
        context.last_remainder = remainders[0]
