import numpy as np

class RobotArmEnv:
    def __init__(self, num_joints=2, target_pos=np.array([0.5, 0.5])):
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints)  # Radians
        self.target_pos = target_pos
        self.state_dim = num_joints * 2  # Joint angles and end-effector position
        self.action_dim = num_joints
        self.max_action = np.pi / 10 # Max change in angle per step

    def _get_end_effector_pos(self):
        # Simple forward kinematics for a 2-joint arm (for demonstration)
        l1 = 0.5 # Length of first link
        l2 = 0.5 # Length of second link
        x = l1 * np.cos(self.joint_angles[0]) + l2 * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y = l1 * np.sin(self.joint_angles[0]) + l2 * np.sin(self.joint_angles[0] + self.joint_angles[1])
        return np.array([x, y])

    def _get_state(self):
        end_effector_pos = self._get_end_effector_pos()
        return np.concatenate([self.joint_angles, end_effector_pos])

    def reset(self):
        self.joint_angles = np.random.uniform(-np.pi, np.pi, self.num_joints)
        return self._get_state()

    def step(self, action):
        # Apply action (change in joint angles)
        action = np.clip(action, -self.max_action, self.max_action)
        self.joint_angles += action
        self.joint_angles = np.clip(self.joint_angles, -2 * np.pi, 2 * np.pi) # Keep angles within reasonable bounds

        current_pos = self._get_end_effector_pos()
        distance_to_target = np.linalg.norm(self.target_pos - current_pos)

        # Reward: negative distance to target (closer is better)
        reward = -distance_to_target

        # Done condition: if close enough to target
        done = distance_to_target < 0.05

        next_state = self._get_state()
        return next_state, reward, done, {}
