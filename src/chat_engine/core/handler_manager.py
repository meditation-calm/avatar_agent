import importlib
import inspect
import os.path
import sys
import time
import weakref

from dataclasses import dataclass, field
from inspect import isclass, isabstract
from types import ModuleType
from typing import Optional, Dict, Tuple

import gradio
from fastapi import FastAPI
from loguru import logger

from src.chat_engine.common.client_handler_base import ClientHandlerBase
from src.chat_engine.common.handler_base import HandlerBaseInfo, HandlerBase
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from src.engine_utils.directory_info import DirectoryInfo


"""
1. 处理器生命周期管理
    加载和注册处理器：通过 initialize 方法根据配置加载处理器模块，并通过 register_handler 方法注册处理器实例
    初始化处理器：调用 load_handlers 方法初始化所有启用的处理器
    销毁处理器：通过 destroy 方法清理所有处理器资源
2. 处理器模块管理
    搜索路径管理：通过 add_search_path 方法管理处理器模块的搜索路径
    动态导入：使用 importlib 动态加载处理器模块
    模块缓存：将已加载的模块存储在 handler_modules 字典中
3. 配置管理
    配置解析：解析并验证每个处理器的配置信息
    并发控制：管理处理器的并发限制设置
    启用状态检查：只加载和使用启用的处理器
4. 处理器分类处理
    客户端处理器识别：识别并特殊处理 ClientHandlerBase 类型的处理器
    优先级排序：根据处理器的优先级进行排序加载
    Web API 集成：为客户端处理器设置 FastAPI 和 Gradio 界面集成
    
工作流程
    初始化时根据配置加载所有启用的处理器模块
    注册处理器并验证其配置
    按优先级顺序加载处理器
    为客户端处理器设置 Web 接口
    在系统关闭时销毁所有处理器
"""


@dataclass
class HandlerRegistry:
    base_info: Optional[HandlerBaseInfo] = field(default=None)
    handler: Optional[HandlerBase] = field(default=None)
    handler_config: Optional[HandlerBaseConfigModel] = field(default=None)


class HandlerManager:
    def __init__(self, engine):
        """
        初始化 HandlerManager 实例：
            创建数据结构存储处理器模块、注册信息和配置
            设置默认并发限制为1
            使用弱引用来引用引擎实例，避免循环引用
        """
        # [handler_module, (module, handler_class)]
        # 存储已加载的处理器模块信息
        self.handler_modules: Dict[str, Tuple[ModuleType, type[HandlerBase]]] = {}
        # [handler_name, handler_registry]
        # 存储已注册的处理器注册信息
        self.handler_registries: Dict[str, HandlerRegistry] = {}
        # [handler_name, handler_config]
        # 存储处理器配置信息
        self.handler_configs: Dict[str, Dict] = {}
        self.concurrent_limit = 1
        self.search_path = []

        self.engine_ref = weakref.ref(engine)

    def initialize(self, engine_config: ChatEngineConfigModel):
        """
        根据引擎配置初始化处理器
            设置并发限制
            添加处理器搜索路径
            存储处理器配置
            加载并注册所有启用的处理器模块
        """
        self.concurrent_limit = engine_config.concurrent_limit
        for search_path in engine_config.handler_search_path:
            self.add_search_path(search_path)
        for handler_name, handler_config in engine_config.handler_configs.items():
            self.handler_configs[handler_name] = handler_config
        logger.info(f"Use handler search path: {self.search_path}")
        for handler_name, raw_config in self.handler_configs.items():
            try:
                handler_config = HandlerBaseConfigModel.model_validate(raw_config)
            except Exception as e:
                logger.error(f"Failed to parse handler config for {handler_name}: {e}")
                continue
            if not handler_config.enabled:
                continue
            if handler_config.module is None:
                logger.warning(f"Handler {handler_name} has no module specified, skipping.")
                continue
            module_path = None
            module_input_path = None
            for search_path in self.search_path:
                find_path = os.path.join(search_path, f"{handler_config.module}.py")
                if os.path.exists(find_path):
                    module_path = find_path
                    module_input_path = handler_config.module.replace("\/", ".").replace("/", ".")
                    break
            if module_path is None:
                logger.error(f"Handler {handler_config.module} not found in search path.")
                raise ValueError(f"Handler {handler_config.module} not found in search path.")
            try:
                logger.info(f"Try to load {module_input_path}")
                module = importlib.import_module(module_input_path)
            except Exception:
                logger.error(f"Failed to import handler module {handler_config.module}")
                raise
            handler_class = None
            for name, obj in inspect.getmembers(module):
                if not isclass(obj):
                    continue
                if isabstract(obj):
                    continue
                if issubclass(obj, HandlerBase):
                    handler_class = obj
                    break
            if handler_class is None:
                logger.error(f"Handler module {handler_config.module} does not contain a HandlerBase subclass.")
                raise ValueError(f"Handler module {handler_config.module} does not contain a HandlerBase subclass.")
            self.handler_modules[handler_config.module] = module, handler_class
            self.register_handler(handler_name, handler_class())

    def add_search_path(self, path: str):
        """
        添加处理器模块搜索路径：
            处理相对路径和绝对路径转换
            验证路径是否为有效目录
            将路径添加到搜索路径列表和系统路径中
        """
        if not os.path.isabs(path):
            if os.path.isdir(path):
                path = os.path.abspath(path)
            else:
                path = os.path.join(DirectoryInfo.get_project_dir(), path)
        if not os.path.isdir(path):
            logger.warning(f"Path {path} is not a directory, it is not added to search path.")
            return
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if path not in self.search_path:
            self.search_path.append(path)
            if path not in sys.path:
                sys.path.append(path)

    def register_handler(self, name: str, handler: HandlerBase):
        """
        注册一个处理器实例：
            创建或获取处理器注册表
            设置处理器根目录和引擎引用
            调用处理器的预注册方法
            获取并验证处理器信息和配置
            完成处理器注册
        """
        registry = self.handler_registries.get(name, None)
        if registry is None:
            registry = HandlerRegistry()
            self.handler_registries[name] = registry
        handler_module = inspect.getmodule(type(handler))
        handler_root = os.path.split(handler_module.__file__)[0]
        handler.handler_root = handler_root
        handler.engine = self.engine_ref
        if registry.base_info is None:
            handler.on_before_register()
            base_info = handler.get_handler_info()
            base_info.name = name
            raw_config = self.handler_configs.get(name, {})
            if not issubclass(base_info.config_model, HandlerBaseConfigModel):
                logger.error(f"Handler {name} provides invalid config model {base_info.config_model}")
                raise ValueError(f"Handler {name} provides invalid config model {base_info.config_model}")
            config: HandlerBaseConfigModel = base_info.config_model.model_validate(raw_config)
            """ 配置独立项控制最大并发数 """
            # config.concurrent_limit = self.concurrent_limit
            registry.base_info = base_info
            registry.handler = handler
            registry.handler_config = config
            logger.info(f"Registered handler {name}({type(handler)}) with config: {config}")

    def load_handlers(self, engine_config: ChatEngineConfigModel,
                      app: Optional[FastAPI] = None,
                      ui: Optional[gradio.blocks.Block] = None,
                      parent_block: Optional[gradio.blocks.Block] = None):
        """
        加载所有启用的处理器：
            获取启用的处理器列表
            区分客户端处理器和其他处理器
            调用每个处理器的加载方法
            为客户端处理器设置Web应用程序接口
        """
        enabled_handlers = self.get_enabled_handler_registries()
        client_handlers = []
        for registry in enabled_handlers:
            if isinstance(registry.handler, ClientHandlerBase):
                client_handlers.append(registry)
            load_start = time.monotonic()
            registry.handler.load(engine_config, registry.handler_config)
            dur_load = time.monotonic() - load_start
            logger.info(f"Handler {registry.base_info.name} loaded in {round(dur_load * 1e3)} milliseconds")
        if app is not None or ui is not None:
            for registry in client_handlers:
                setup_start = time.monotonic()
                registry.handler.on_setup_app(app, ui, parent_block)
                dur_setup = time.monotonic() - setup_start
                logger.info(
                    f"Setup client handler {registry.base_info.name} loaded in {round(dur_setup * 1e3)} milliseconds")

    def get_enabled_handler_registries(self, order_by_priority=True):
        """
        获取所有启用的处理器注册信息：
            过滤出启用的处理器
            可选择按优先级排序
        """
        result = []
        for handler_name, registry in self.handler_registries.items():
            if registry.handler is None or registry.handler_config is None:
                continue
            if not registry.handler_config.enabled:
                continue
            result.append(registry)
        if order_by_priority:
            result.sort(key=lambda x: x.base_info.load_priority)
        return result

    def find_client_handler(self, handler):
        """
        查找指定的客户端处理器：
            在注册表中查找匹配的客户端处理器
            返回对应的注册信息
        """
        if handler is None:
            return None
        for handler_name, registry in self.handler_registries.items():
            if registry.handler is None or registry.handler_config is None:
                continue
            if not registry.handler_config.enabled:
                continue
            if isinstance(registry.handler, ClientHandlerBase) and registry.handler is handler:
                return registry

    def destroy(self):
        """
        销毁所有启用的处理器：
            遍历所有处理器注册信息
            对启用的处理器调用销毁方法
            记录销毁过程日志
        """
        for handler_name, registry in self.handler_registries.items():
            if registry.handler is None or registry.handler_config is None:
                continue
            if not registry.handler_config.enabled:
                continue
            logger.info(f"Destroying handler {handler_name}")
            registry.handler.destroy()
            logger.info(f"Handler {handler_name} destroyed")
