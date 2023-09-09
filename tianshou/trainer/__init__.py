"""Trainer package."""

from tianshou.trainer.base import BaseTrainer
from tian.trainer.offpolicy_of4on_dpc import (
    OnlinePolicyExperienceCollectionSetting,
    OffpolicyDualpolicyCollectionTrainer,
    offpolicy_dualpolicycollection_trainer,
    offpolicy_dualpolicycollection_trainer_iter,
)
from tianshou.trainer.offline import (
    OfflineTrainer,
    offline_trainer,
    offline_trainer_iter,
)
from tianshou.trainer.offpolicy import (
    OffpolicyTrainer,
    offpolicy_trainer,
    offpolicy_trainer_iter,
)
from tianshou.trainer.onpolicy import (
    OnpolicyTrainer,
    onpolicy_trainer,
    onpolicy_trainer_iter,
)
from tianshou.trainer.utils import gather_info, test_episode

__all__ = [
    "BaseTrainer",
    "OnlinePolicyExperienceCollectionSetting",
    "offpolicy_dualpolicycollection_trainer",
    "offpolicy_dualpolicycollection_trainer_iter",
    "OffpolicyDualpolicyCollectionTrainer",
    "offpolicy_trainer",
    "offpolicy_trainer_iter",
    "OffpolicyTrainer",
    "onpolicy_trainer",
    "onpolicy_trainer_iter",
    "OnpolicyTrainer",
    "offline_trainer",
    "offline_trainer_iter",
    "OfflineTrainer",
    "test_episode",
    "gather_info",
]
