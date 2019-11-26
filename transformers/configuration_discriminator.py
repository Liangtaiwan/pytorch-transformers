""" Discriminator model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from .configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


class DiscriminatorConfig(PretrainedConfig):
    def __init__(self,
                 config_json_file=None,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(config_json_file, str):
            with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            raise ValueError("First argument must be the path to a discriminators config file (str)")
