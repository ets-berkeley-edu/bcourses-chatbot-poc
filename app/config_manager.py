"""
Copyright ©2024. The Regents of the University of California (Regents). All Rights Reserved.

Permission to use, copy, modify, and distribute this software and its documentation
for educational, research, and not-for-profit purposes, without fee and without a
signed licensing agreement, is hereby granted, provided that the above copyright
notice, this paragraph and the following two paragraphs appear in all copies,
modifications, and distributions.

Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue,
Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED
"AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.
"""

import importlib.util
import os
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging

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
        try:
            if importlib.util.find_spec(config_path) is not None:
                module = importlib.import_module(config_path)
                cls.config.update(vars(module.Config))
            else:
                logging.warning(f"Configuration module not found: {config_path}")
        except Exception as e:
            logging.error(f"Error loading module config {config_path}: {e}")

    @classmethod
    def load_local_config(cls, config_name):
        configs_location = os.environ.get("APP_LOCAL_CONFIGS", "../config")
        config_path = os.path.join(configs_location, config_name)
        try:
            if os.path.exists(config_path):
                spec = importlib.util.spec_from_file_location("local_config", config_path)
                local_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_config)
                cls.config.update(vars(local_config.Config))
            else:
                logging.warning(f"Local configuration file not found: {config_path}")
        except Exception as e:
            logging.error(f"Error loading local config {config_path}: {e}")