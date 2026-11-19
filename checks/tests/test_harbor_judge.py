"""
Tests of the judge for Harbor trials (skyrl_extensions/harbor_judge) against the fake OpenAI-compatible server of
test_judge.py: no model, no GPU, no sandbox. SkyRL's own trial loop is replaced by a stub that returns finished
trajectories. Needs SkyRL's Harbor integration, openai and pydantic (the container has them; skipped elsewhere):
    ./check.sh selftest -k harbor_judge
"""
import asyncio
import json
import types

import pytest

pytest.importorskip("openai")
hg = pytest.importorskip("examples.train_integrations.harbor.harbor_generator")

from checks.tests.test_judge import MODEL, FakeJudge  # noqa: E402
from skyrl_extensions.harbor_judge.generator import JudgedHarborGenerator  # noqa: E402
from skyrl_extensions.judge import env as judge_env  # noqa: E402

INSTRUCTION = "Write the words hello world into /app/out.txt."
SOLUTION = "#!/bin/bash\necho hello world > /app/out.txt\n"


class CharTokenizer:
    """One token per character: decode is exact, so the test can say what the judge must see."""

    @staticmethod
    def encode(text):
        return [ord(c) for c in text]

    @staticmethod
    def decode(ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def rollout(turns, first_prompt="<sys>task prompt</sys>"):
    """rollout_details[0] of a linear chat: turns = [(assistant reply, terminal output shown after it), ...]."""
    tok, prompts, completions, convo = CharTokenizer, [], [], first_prompt
    for reply, observation in turns:
        prompts.append(tok.encode(convo))
        completions.append(tok.encode(reply))
        convo += reply + observation
    return {"prompt_token_ids": prompts, "completion_token_ids": completions, "logprobs": [[0.0] * len(c) for c in completions]}


def trajectory(turns, reward=1.0, stop_reason="complete"):
    return types.SimpleNamespace(rollout_details=[rollout(turns)], reward=reward, stop_reason=stop_reason)


@pytest.fixture
def judge(monkeypatch):
    fake = FakeJudge()
    monkeypatch.setattr(judge_env, "_client", None)
    for k, v in {"JUDGE_NAME": "harbor_task", "JUDGE_SERVED_NAME": MODEL, "JUDGE_URL": fake.url, "JUDGE_RETRIES": "1",
                 "JUDGE_MAX_TOKENS": "256", "JUDGE_MAX_CHARS": "60000"}.items():
        monkeypatch.setenv(k, v)
    fake.reply = json.dumps({"reasoning": "r", "task_completed": True})
    yield fake
    fake.server.shutdown()


@pytest.fixture
def task(tmp_path):
    t = tmp_path / "my-dataset" / "hello-task"
    (t / "solution").mkdir(parents=True)
    (t / "instruction.md").write_text(INSTRUCTION)
    (t / "solution" / "solve.sh").write_text(SOLUTION)
    return str(t)


@pytest.fixture
def gen(judge):
    """A JudgedHarborGenerator without SkyRL's __init__ (no engines): only what the judge part uses."""
    g = object.__new__(JudgedHarborGenerator)
    g.tokenizer, g.judge_client, g._judge_metrics, g._nonlinear_warned = CharTokenizer, judge_env.connect(), [], False
    return g


TURNS = [("<think>plan it</think>I will write the file.\n<cmd>echo hello world > /app/out.txt</cmd>", "\n<obs>$ </obs>"),
         ("<cmd>cat /app/out.txt</cmd>", "\n<obs>hello world</obs>"),
         ("<think>done</think>The file contains hello world. Task complete.", "")]


def test_the_judge_sees_instruction_solution_and_session(gen, judge, task):
    reward, m = gen._judge(task, trajectory(TURNS, reward=1.0))
    assert reward == 1.0 and m["judge_task_completed"] == 1.0 and m["judge_unparsed"] == 0.0
    assert m["rule_reward"] == 1.0 and m["judge_rule_agree"] == 1.0 and m["reward_my-dataset"] == 1.0
    req = judge.requests[0]
    text = req["messages"][0]["content"]
    assert req["response_format"]["json_schema"]["schema"]["required"] == ["reasoning", "task_completed"]
    assert INSTRUCTION in text and SOLUTION.strip() in text
    session = text[text.rindex("<response>\n"):text.rindex("</response>")]   # the fence (the note names the tag too)
    # everything after the task prompt, replies and terminal output, without the thinking
    assert "<sys>" not in session and "<think>" not in session and "plan it" not in session
    assert "I will write the file." in session and "<obs>hello world</obs>" in session and "Task complete." in session


def test_judge_replaces_the_verifier_reward_and_disagreement_is_logged(gen, judge, task):
    judge.reply = json.dumps({"reasoning": "r", "task_completed": False})
    reward, m = gen._judge(task, trajectory(TURNS, reward=1.0))
    assert reward == 0.0 and m["rule_reward"] == 1.0 and m["judge_rule_agree"] == 0.0


def test_unparsed_reply_scores_zero_and_is_counted(gen, judge, task, monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setattr(judge_env, "_client", None)
    gen.judge_client = judge_env.connect()
    judge.mode = "garbage"
    reward, m = gen._judge(task, trajectory(TURNS))
    assert reward == 0.0 and m["judge_unparsed"] == 1.0
    assert json.loads((tmp_path / "judge_unparsed.jsonl").read_text().splitlines()[0])["data_source"] == "my-dataset"


def test_dead_judge_raises_instead_of_scoring_zero(gen, judge, task):
    judge.mode = "down"
    with pytest.raises(Exception):
        gen._judge(task, trajectory(TURNS))


def test_a_session_that_is_only_thinking_is_not_sent(gen, judge, task):
    reward, m = gen._judge(task, trajectory([("<think>hmm, never finished", "")]))
    assert reward == 0.0 and m["judge_skipped"] == 1.0 and judge.requests == []


def test_task_without_solution_and_nonlinear_chat(gen, judge, task, tmp_path):
    import os
    os.remove(os.path.join(task, "solution", "solve.sh"))
    out = trajectory(TURNS)
    out.rollout_details[0]["prompt_token_ids"][-1] = CharTokenizer.encode("<sys>summarized</sys>")   # not a prefix
    gen._judge(task, out)
    text = judge.requests[0]["messages"][0]["content"]
    assert "<solution>" not in text and "<sys>" not in text
    assert "I will write the file." in text and "<obs>" not in text   # replies only


def test_only_finished_trials_are_judged_and_metrics_are_aggregated(gen, judge, task, monkeypatch):
    outs = {"a": trajectory(TURNS, reward=0.0), "b": trajectory(TURNS, reward=0.0, stop_reason="context_length"),
            "c": types.SimpleNamespace(rollout_details=None, reward=0.0, stop_reason="error")}

    async def fake_loop(self, prompt, trajectory_id, cache_salt=None):
        return outs[trajectory_id]

    async def fake_generate(self, input_batch, disable_tqdm=False):
        res = [await self._harbor_agent_loop(prompt=task, trajectory_id=t) for t in input_batch["trajectory_ids"]]
        return {"rewards": [r.reward for r in res], "rollout_metrics": {"generate/avg_num_turns": 3.0}}

    monkeypatch.setattr(hg.HarborGenerator, "_harbor_agent_loop", fake_loop)
    monkeypatch.setattr(hg.HarborGenerator, "generate", fake_generate)
    out = asyncio.run(gen.generate({"prompts": [task] * 3, "trajectory_ids": ["a", "b", "c"]}))
    assert out["rewards"] == [1.0, 0.0, 0.0]          # a judged (verifier said 0), b context length, c masked
    assert len(judge.requests) == 1
    m = out["rollout_metrics"]
    assert m["generate/avg_num_turns"] == 3.0 and m["generate/judged_trajectories"] == 1
    assert m["generate/judge_task_completed"] == 1.0 and m["generate/judge_rule_agree"] == 0.0


def test_single_completion_judges_are_refused_for_harbor():
    from skyrl_extensions.judge.judges import JUDGES
    assert [n for n, j in JUDGES.items() if getattr(j, "harbor", False)] == ["harbor_task"]
