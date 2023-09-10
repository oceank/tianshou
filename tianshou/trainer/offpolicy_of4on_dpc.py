from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import numpy as np

from tianshou.data import Collector, Of4OnCollector
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger
from tianshou.trainer.offpolicy import OffpolicyTrainer


class OnlinePolicyExperienceCollectionSetting:
    def __init__(self, setting_type: str = "Fixed", setting_value: Union[float, List] = 1.0):
        self.setting_type = setting_type
        self.setting_value = setting_value

        if self.setting_type == "Fixed":
            # self.setting_value is the value of the fixed ratio
            assert isinstance(self.setting_value, float), f"Expected a float setting value for the Fixed ratio setting. But got {self.setting_value}"
        elif self.setting_type == "Scheduling":
            # self.setting_value: [start_ratio, end_ratio]
            assert isinstance(self.setting_value, List), f"Expected a list of setting value for the Scheduling ratio setting. But got {self.setting_value}"
            assert (len(self.setting_value)==2 and isinstance(self.setting_value[0], float) and isinstance(self.setting_value[1], float)), f"Expected the setting value to be a list of two float numbers for the Scheduling ratio setting. But got {self.setting_value}"
            assert ((self.setting_value[0]>=0 and self.setting_value[1]<=1) and (self.setting_value[0] and self.setting_value[1]<=1)), f"Expected the setting value contains a list of two float numbers ranging from 0 to 1 for the Scheduling ratio setting. But got {self.setting_value}"
        elif self.setting_type == "Adaptive":
            # self.setting_value is the test performance of the offline policy
            assert isinstance(self.setting_value, float), f"Expected the setting value to be a float number. But got {self.setting_value}"
        else:
            assert False, f"Unexpected setting type, {self.setting_type}"

    def cal_online_policy_experience_collection_ratio(
            self,
            training_progress_in_percent: Optional[float] = None,
            online_policy_test_performance: Optional[float] = None,
            ) -> float:
        if self.setting_type == "Fixed":
            return self.setting_value
        elif self.setting_type == "Scheduling":
            return self.setting_value[0] + training_progress_in_percent*(self.setting_value[1]-self.setting_value[0])
        elif self.setting_type == "Adaptive":
            performances = np.array([online_policy_test_performance, self.setting_value])
            probs = self.softmax(performances - np.max(performances))
            return probs[0]
        else:
            assert False, f"Fail to calculate the ratio of experience collection by the online policy due to the Unexpected setting type, {self.setting_type}"

    def softmax(self, x):
        f = np.exp(x - np.max(x)) # shift the highest value
        return f / f.sum(axis=0)

    def __str__(self):
        return f"setting_type ({self.setting_type}), setting_value ({self.setting_value})"

class OffpolicyDualpolicyCollectionTrainer(OffpolicyTrainer):
    """Create an iterator wrapper for off-policy training procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
        set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would
        collect before the network update, i.e., trainer will collect
        "step_per_collect" transitions and do some policy network update repeatedly
        in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param int/float update_per_step: the number of times the policy network would
        be updated per transition after (step_per_collect) transitions are
        collected, e.g., if update_per_step set to 0.3, and step_per_collect is 256
        , policy will be updated round(256 * 0.3 = 76.8) = 77 times after 256
        transitions are collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) ->  None``. It was ``save_fn`` previously.
    :param function save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata
        from existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature
        ``f(rewards: np.ndarray with shape (num_episode, agent_num)) ->
        np.ndarray with shape (num_episode,)``, used in multi-agent RL. We need to
        return a single scalar for each episode's result to monitor training in the
        multi-agent RL setting. This function specifies what is the desired metric,
        e.g., the reward of agent 1 or the average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training.
        Default to True.
    :param bool test_in_train: whether to test in the training phase.
        Default to True.
    """

    __doc__ = BaseTrainer.gen_doc("offpolicyOf4Ondualcollection") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Of4OnCollector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        step_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        update_per_step: Union[int, float] = 1,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        online_policy_experience_collection_setting: OnlinePolicyExperienceCollectionSetting = OnlinePolicyExperienceCollectionSetting("Fixed", 1.0),
        **kwargs: Any,
    ):
        self.online_policy_experience_collection_setting = online_policy_experience_collection_setting

        super().__init__(
            policy,
            train_collector,
            test_collector,
            max_epoch,
            step_per_epoch,
            step_per_collect,
            episode_per_test,
            batch_size,
            update_per_step = update_per_step,
            train_fn = train_fn,
            test_fn = test_fn,
            stop_fn = stop_fn,
            save_best_fn = save_best_fn,
            save_checkpoint_fn = save_checkpoint_fn,
            resume_from_log = resume_from_log,
            reward_metric = reward_metric,
            logger = logger,
            verbose = verbose,
            show_progress = show_progress,
            test_in_train = test_in_train,
            **kwargs,
        )

        self.num_epochs_to_run = self.max_epoch - self.epoch

    def __iter__(self):  # type: ignore
        super().reset()
        # self.epoch equals 0 or the number of epochs used to train the resumed policy,
        # so, (self.max_epoch - self.epoch) is the number of epochs to run here
        self.num_epochs_to_run = self.max_epoch - self.epoch
        return self

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        # set the ratio of experience collection by online policy (i.e., the data member, policy)
        assert self.train_collector is not None

        ratio = self.online_policy_experience_collection_setting.cal_online_policy_experience_collection_ratio(
                float((1+self.iter_num)/self.num_epochs_to_run), # training progresess in percentage
                self.recent_reward, # the test performance of the recent online policy
        )
        if self.online_policy_experience_collection_setting.setting_type == "Adaptive":
            print(f"[Online Policy Experience Collection Ratioi - Adaptive] online policy's performance ({self.recent_reward}), offline policy's performnace ({self.online_policy_experience_collection_setting.setting_value}) => ratio ({ratio})")
        self.train_collector.set_online_policy_collecting_ratio(ratio)

        return super().__next__() 


def offpolicy_dualpolicycollection_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OffPolicyTrainer run method.

    It is identical to ``OffpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OffpolicyDualpolicyCollectionTrainer(*args, **kwargs).run()


offpolicy_dualpolicycollection_trainer_iter = OffpolicyDualpolicyCollectionTrainer
