"""Instantiate a temporal consistent policy flow model."""

import logging

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import (
    LeRobotDatasetMetadata,
    make_policy,
)
from lerobot.utils.import_utils import register_third_party_plugins
from torch.utils.data import DataLoader

register_third_party_plugins()

PreTrainedConfig.get_known_choices().keys()


logging.basicConfig(level=logging.INFO)

# %%
policy_name = "ditflow"
dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
policy_features = dataset_to_policy_features(dataset_metadata.features)
print(policy_features)

# %%
default_ditflow_config = PreTrainedConfig.get_choice_class(policy_name)()

# %%
policy = make_policy(default_ditflow_config, ds_meta=dataset_metadata)
policy.config.do_consistent_flow = True
sum(p.numel() for p in policy.parameters() if p.requires_grad)
policy.to("cpu")


# %%
print("Instantiate DIT policy model successfully!")

policy.config.input_features

# %%
for name, child in policy.named_children():
    print(name, "->")
    for grandchild_name, grandchild in child.named_children():
        print("   ", grandchild_name, "->")
        for great_grandchild_name, _ in grandchild.named_children():
            print("       ", great_grandchild_name)

# %%

dataset = LeRobotDataset("lerobot/pusht")

# %%
