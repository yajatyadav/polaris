import csv
import os
import time

import numpy as np
from openpi_client import websocket_client_policy, image_tools
from polaris.policy.abstract_client import InferenceClient, PolicyArgs


# Joint Position Client for DROID
"""
This client was modified to also hold a self.num_candidates attribute. Everything is same, just the policy-server call is now .infer(obs, num_candidates=self.num_candidates)
The policy server handles sampling multiple candidates and returning the best one to this inferenceclient.
"""
@InferenceClient.register(client_name="DroidJointPos")
class DroidJointPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DroidJointPosClient")

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon
        self.num_candidates = args.num_candidates

        # Classifier metrics logging
        self.log_dir = args.log_dir
        self._env_step = 0
        self._infer_call = 0
        self._rollout_idx = -1
        self._csv_writer = None
        self._csv_file = None
        assert args.log_interval % self.open_loop_horizon == 0, (
            f"log_interval ({args.log_interval}) must be divisible by open_loop_horizon ({self.open_loop_horizon})"
        )
        self._log_every_n_infers = args.log_interval // self.open_loop_horizon

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def _open_metrics_csv(self):
        """Open a new classifier_metrics_<rollout>.csv file."""
        if self._csv_file is not None:
            self._csv_file.close()
        self._csv_writer = None
        self._csv_file = None
        self._csv_headers = None
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            csv_path = os.path.join(self.log_dir, f"classifier_metrics_{self._rollout_idx}.csv")
            self._csv_file = open(csv_path, "w", newline="")

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self._env_step = 0
        self._infer_call = 0
        self._rollout_idx += 1
        self._open_metrics_csv()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the policy in a server-client setup
        """
        both = None
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)

            self.actions_from_chunk_completed = 0
            exterior_image = image_tools.resize_with_pad(
                curr_obs["right_image"], 224, 224
            )
            wrist_image = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            request_data = {
                "observation/exterior_image_1_left": exterior_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            if self.num_candidates is not None:
                request_data["num_candidates"] = self.num_candidates
            server_response = self.client.infer(request_data)
            self.pred_action_chunk = server_response["actions"]
            self._log_classifier_metrics(server_response)
            both = np.concatenate([exterior_image, wrist_image], axis=1)

        if return_viz and both is None:
            curr_obs = self._extract_observation(obs)
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                    image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        return action, both

    @staticmethod
    def _format_value(v):
        """Convert a value to a CSV-friendly string."""
        if v is None:
            return ""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return str(v)
        return v

    def _log_classifier_metrics(self, server_response: dict):
        """Log all fields from server_response to CSV.
        Never raises — logging failures must not kill the eval run."""
        env_step = self._infer_call * self.open_loop_horizon
        self._infer_call += 1
        if self._csv_file is None or self._infer_call % self._log_every_n_infers != 1:
            return
        try:
            # Skip "actions" since it's large and already consumed
            response_keys = sorted(k for k in server_response if k != "actions")

            # Write header on first logged row (deferred so we know the keys)
            if self._csv_headers is None:
                self._csv_headers = ["env_step", "timestamp"] + response_keys
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(self._csv_headers)

            row = [env_step, time.time()]
            for k in response_keys:
                row.append(self._format_value(server_response.get(k)))
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception as e:
            print(f"[DroidJointPosClient] classifier metrics logging failed (env_step {env_step}): {e}")

    def _extract_observation(self, obs_dict):
        # Assign images
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
