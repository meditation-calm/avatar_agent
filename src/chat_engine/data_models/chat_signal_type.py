from enum import Enum


class ChatSignalType(str, Enum):
    """
    系统中可能发生的信号类型：
        BEGIN = "begin"：开始信号，表示某个操作或会话的开始
        END = "end"：结束信号，表示某个操作或会话的结束
        INTERRUPT = "interrupt"：中断信号，表示当前操作被中
    """
    # START = "start"
    BEGIN = "begin"
    END = "end"
    # CANCEL = "cancel"
    INTERRUPT = "interrupt"
    # RESET = "reset"
    # ERROR = "error"
    # STOP = "stop"


class ChatSignalSourceType(str, Enum):
    """
    信号的来源类型：
        CLIENT = "client"：来自客户端的信号
    """
    CLIENT = "client"
    # LOGIC = "logic"
    # HANDLER = "handler"
    # ENGINE = "engine"
