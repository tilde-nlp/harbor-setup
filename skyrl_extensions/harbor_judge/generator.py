"""
Harbor trials graded by a judge model: SkyRL's HarborGenerator (examples/train_integrations/harbor/
harbor_generator.py), whose reward, the task verifier's, is replaced by the verdict of the judge (the config's
`judge:` section, a judge with `harbor = True` in skyrl_extensions/judge/judges.py) on the agent's session.
Everything else (trials, retries, masking, step-wise output) is SkyRL's.

What the judge gets, r as for the single-step judge (judges.py):
    prompt      [{"role": "user", "content": the task's instruction.md}]
    completion  the agent's session after the task prompt: its replies and the terminal output it was shown,
                decoded from the policy's own tokens. The last turn's prompt + completion is the whole
                conversation (the chat is linear: step-wise training rules out summarization). <think> blocks
                are removed: the judge grades what the agent did, not its thinking.
    shown       completion cut to judge.max_chars (head 2/3 + tail 1/3)
    extras      task (the task dir), task_name, data_source (the dataset dir's name), solution (the task's
                solution/solve.sh, if any), verifier_reward (the task's own test result)
The judge does not see the container: what the agent left there counts only as far as the session shows it.

Which trials are judged: those that finished ("complete"). Failed and timed-out trials stay masked, and
context-length ones keep reward 0, as in SkyRL. The tasks still need a verifier: SkyRL retries and then masks a
trial without a verifier result. For a task without tests, a tests/test.sh that writes 0 to
/logs/verifier/reward.txt is enough (its reward is only logged).

A judge that fails (after the client's retries) ends the run, as in the single-step judge: a dead judge must not
look like "every trial failed".

Logged per step, means over the judged trajectories (generate/judge_*): judge_unparsed (reply did not fit the
schema: reward 0, a line in $RUN_DIR/judge_unparsed.jsonl), judge_skipped (decided without the judge: empty
session), judge_seconds, judge_<field> for every bool / number the judge answered, rule_reward (the verifier's
reward) and judge_rule_agree (judge reward > 0 == verifier reward > 0), reward_<dataset> (per data_source).
"""
import asyncio
import os
import re
import time
from collections import defaultdict

from loguru import logger

from examples.train_integrations.harbor.harbor_generator import HarborGenerator
from skyrl_extensions.judge import env as judge_env

THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)
SOLUTION_MAX_CHARS = 8000


def read_text(path, max_chars=None):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    return text if max_chars is None else judge_env.truncate(text, max_chars)


class JudgedHarborGenerator(HarborGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.judge_client = judge_env.connect()
        self._judge_metrics = []
        self._nonlinear_warned = False

    async def generate(self, input_batch, disable_tqdm=False):
        self._judge_metrics = []
        output = await super().generate(input_batch, disable_tqdm=disable_tqdm)
        sums, counts = defaultdict(float), defaultdict(int)
        for m in self._judge_metrics:
            for k, v in m.items():
                sums[k] += v
                counts[k] += 1
        if output.get("rollout_metrics") is None:
            output["rollout_metrics"] = {}
        output["rollout_metrics"].update({"generate/" + k: sums[k] / counts[k] for k in sums})
        output["rollout_metrics"]["generate/judged_trajectories"] = len(self._judge_metrics)
        return output

    async def _harbor_agent_loop(self, prompt, trajectory_id, cache_salt=None):
        out = await super()._harbor_agent_loop(prompt=prompt, trajectory_id=trajectory_id, cache_salt=cache_salt)
        if out.rollout_details is None or out.stop_reason != "complete":
            return out   # masked, or context length (reward 0): not judged
        reward, metrics = await asyncio.to_thread(self._judge, prompt, out)
        out.reward = reward
        self._judge_metrics.append(metrics)
        return out

    def session_text(self, rollout_detail):
        """The conversation after the task prompt, as the policy saw and wrote it, without <think> blocks."""
        prompts, completions = rollout_detail["prompt_token_ids"], rollout_detail["completion_token_ids"]
        first, full = list(prompts[0]), list(prompts[-1]) + list(completions[-1])
        if full[:len(first)] == first:
            text = self.tokenizer.decode(full[len(first):], skip_special_tokens=False)
        else:   # not one linear chat: the agent's replies only
            if not self._nonlinear_warned:
                logger.warning("harbor judge: a trajectory is not one linear chat; the judge sees the agent's replies only")
                self._nonlinear_warned = True
            text = "\n\n".join(self.tokenizer.decode(c, skip_special_tokens=False) for c in completions)
        return THINK.sub("", text)

    def _judge(self, task_dir, out):
        """-> (reward, metrics) for one finished trajectory. Runs in a thread: the openai client blocks."""
        client = self.judge_client
        session = self.session_text(out.rollout_details[0])
        extras = {
            "task": task_dir,
            "task_name": os.path.basename(task_dir.rstrip("/")),
            "data_source": os.path.basename(os.path.dirname(task_dir.rstrip("/"))),
            "solution": read_text(os.path.join(task_dir, "solution", "solve.sh"), SOLUTION_MAX_CHARS),
            "verifier_reward": out.reward,   # SkyRL's reward so far: the verifier's
        }
        r = {"prompt": [{"role": "user", "content": read_text(os.path.join(task_dir, "instruction.md")) or ""}],
             "completion": session, "shown": judge_env.truncate(session, client.max_chars), "extras": extras}
        judge = client.judge_for(extras)
        metrics = {"judge_unparsed": 0.0, "judge_skipped": 0.0}

        reward = 0.0 if not session.strip() else judge.precheck(r)
        if reward is not None:
            metrics["judge_skipped"] = 1.0
        else:
            start = time.monotonic()
            scores, _ = client.ask(r, judge)
            metrics["judge_seconds"] = time.monotonic() - start
            if scores is None:
                reward, metrics["judge_unparsed"] = 0.0, 1.0
            else:
                reward = float(judge.reward(scores, r))
                metrics.update({"judge_" + k: float(v) for k, v in scores.items() if isinstance(v, (bool, int, float))})
        metrics["judge_reward"] = reward
        rule = judge.rule(r)
        if rule is not None:
            metrics["rule_reward"] = float(rule)
            metrics["judge_rule_agree"] = float((reward > 0) == (rule > 0))
        metrics["reward_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", extras["data_source"] or "tasks")] = reward
        return reward, metrics
