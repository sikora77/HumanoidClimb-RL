

class State:
    def __init__(self, climber):
        self.climber = climber
        self.joint_positions = {}
        self.joint_velocities = {}
        self.base_position = None
        self.base_orientation = None

    def get_current_state(self, client, robot_id):
        _pos, _ori = client.getBasePositionAndOrientation(robot_id)
