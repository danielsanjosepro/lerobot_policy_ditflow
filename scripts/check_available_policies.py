"""A script to check and print all available pre-trained policies in the lerobot library."""

import lerobot
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

print(
    f"The available policies are: {list[PreTrainedConfig.get_known_choices().keys()]}"
)
