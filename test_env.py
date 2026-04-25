from src.environment import AdPolicyEnvironment
from src.models import AdAction


def run_episode(task_id, actions):
    env = AdPolicyEnvironment()
    obs = env.reset(task_id=task_id)

    for act in actions:
        obs = env.step(
            AdAction(
                action_type=act,
                reasoning="smoke test",
                violation_category="NONE",
            )
        )
        if obs.done:
            break

    return env, obs


if __name__ == "__main__":
    env1, obs1 = run_episode(
        "task_1_healthcare",
        [
            "query_regulations",
            "analyze_image",
            "check_advertiser_history",
            "submit_audit",
            "reject",
        ],
    )

    assert len(env1.trace) >= 4, f"Trace too short: {len(env1.trace)}"
    assert isinstance(env1.total_reward, float), "Reward is not numeric"
    assert all("summary" in t["result"] for t in env1.trace), "Bad trace format"

    env2, obs2 = run_episode(
        "task_10_failure",
        [
            "query_regulations",
            "query_regulations",
            "check_advertiser_history",
            "submit_audit",
            "reject",
        ],
    )

    assert len(env2.trace) >= 2, f"Failure trace too short: {len(env2.trace)}"
    assert any("API failure" in t["result"]["summary"] for t in env2.trace), (
        "Failure case did not trigger"
    )

    print("STEP 7 SMOKE TEST PASSED")
    print("\nTRACE 1:")
    for row in env1.trace:
        print(row)

    print("\nTRACE 2:")
    for row in env2.trace:
        print(row)

    print("\nTOTAL REWARD 1:", env1.total_reward)
    print("TOTAL REWARD 2:", env2.total_reward)