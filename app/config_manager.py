import importlib.util
import os


class ConfigManager:
    config = {}

    @classmethod
    def initialize(cls):
        cls.load_configs()

    @classmethod
    def load_configs(cls):
        cls.load_module_config("default")
        app_env = os.environ.get("APP_ENV", "development")
        cls.load_module_config(app_env)
        cls.load_local_config(f"{app_env}-local.py")
        cls.config["APP_ENV"] = app_env

    @classmethod
    def load_module_config(cls, config_name):
        config_path = f"config.{config_name}"
        if importlib.util.find_spec(config_path) is not None:
            module = importlib.import_module(config_path)
            cls.config.update(vars(module.Config))

    @classmethod
    def load_local_config(cls, config_name):
        configs_location = os.environ.get("APP_LOCAL_CONFIGS", "../config")
        config_path = os.path.join(configs_location, config_name)
        if os.path.exists(config_path):
            spec = importlib.util.spec_from_file_location("local_config", config_path)
            local_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_config)
            cls.config.update(vars(local_config.Config))