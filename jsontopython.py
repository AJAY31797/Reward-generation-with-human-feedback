import torch
import numpy as np

def apply_shape(variable,shape):
    if shape == "linear":
        return variable
    elif shape == "quadratic":
        return variable ** 2
    elif shape == "sqrt":
        return np.sqrt(abs(variable))
    elif shape == "log":
        return np.log(abs(variable) + 1e-8)
    else:
        return variable

def compile_reward(spec):
    """
    Compiles a RewardSpec JSON into a callable reward function.
    """

    # -------- Validation --------

    apply_terminal_only_if_complete = spec.get(
        "apply_terminal_only_if_complete", True
    )

    step_terms = spec.get("step_terms", [])
    terminal_terms = spec.get("terminal_terms", [])

    # -------- Reward function --------
    def reward_fn(
        incremental_cost,
        total_episode_cost,
        timestep,
        action,
        number_of_started_tasks,
        deltat,
        current_day,
        number_of_finished_tasks,
        all_completion_status,
        weight_vector,
        n_elements
    ):
        # Map variable names to values
        values = {
            "incremental_cost": incremental_cost,
            "total_episode_cost": total_episode_cost,
            "timestep": timestep,
            "number_of_started_tasks": number_of_started_tasks,
            "deltat": deltat,
            "current_day": current_day,
            "number_of_finished_tasks": number_of_finished_tasks,
            "all_completion_status": all_completion_status,
            "n_elements": n_elements
        }

        time_reward = 0.0
        cost_reward = 0.0

        # ----- Step terms -----
        for term in step_terms:
            x = values[term["source"]]
            shaped = apply_shape(x, term["shape"])
            contribution = term["weight"] * shaped

            if term["channel"] == "time":
                time_reward += contribution
            elif term["channel"] == "cost":
                cost_reward += contribution
            elif term["channel"] == "both":
                time_reward += contribution
                cost_reward += contribution

        # ----- Terminal terms -----
        if (not apply_terminal_only_if_complete) or all_completion_status:
            for term in terminal_terms:
                x = values[term["source"]]
                shaped = apply_shape(x, term["shape"])
                contribution = term["weight"] * shaped
                if term["channel"] == "time":
                    time_reward += contribution
                elif term["channel"] == "cost":
                    cost_reward += contribution
                elif term["channel"] == "both":
                    time_reward += contribution
                    cost_reward += contribution

        # Return as torch tensor (matches PPO storage)
        reward = np.array([time_reward, cost_reward],dtype=float)
        return reward, total_episode_cost, reward

    return reward_fn
