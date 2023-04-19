from typing import Any, Iterable


class Watcher:
    def __init__(self, log_dir: str, experiment: str, name: str, experiment_dir: str):
        raise NotImplementedError()

    def add_scalar(self, name: str, value: float, step: int, context: Any):
        raise NotImplementedError()
    
    def add_distribution(self, name: str, data: Iterable, step: int, context: Any):
        raise NotImplementedError()
    
    def add_figure(self, name: str, figure, step: int, context: Any):
        raise NotImplementedError()

    def add_text(self, name: str, value: str, step: int, context: Any):
        raise NotImplementedError()
    
    def add_result(self, result: dict[str, Any]):
        raise NotImplementedError()
    
    def add_config(self, config: dict[str, Any]):
        raise NotImplementedError()

    def add_tag(self, tag: str):
        raise NotImplementedError()

    def info(self):
        return {}

    def close(self):
        raise NotImplementedError()
