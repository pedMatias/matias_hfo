from hfo import IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, \
    OUT_OF_TIME, DRIBBLE, SHOOT, MOVE


def simple_reward(game_status: int) -> int:
    if game_status == IN_GAME:
        return -1
    elif game_status == GOAL:
        return 1000
    elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
        return -500
    else:
        return 0


def reward_v0(game_status: int, can_kick: bool, hfo_action: int) -> int:
    goal_reward = 10000
    lose_game_discount = - goal_reward
    time_step_discount = - 3
    right_action_reward = 2
    reward = 0
    if (can_kick and hfo_action in [DRIBBLE, SHOOT]) or \
            (not can_kick and hfo_action == MOVE):
        reward += right_action_reward
    if game_status == GOAL:
        return goal_reward + reward
    elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS]:
        return lose_game_discount + reward
    else:
        return time_step_discount + reward
