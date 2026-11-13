"""
skyrl-gym env "judge": single-step, the reward of a completion comes from a judge model behind an
OpenAI-compatible API (a vLLM server on the job's judge node, launch/job.sbatch role `judge`).

Modelled on SkyRL's examples/train/llm_as_a_judge/llm_judge_env.py. What is different, and why:
    judge failures are not rewards   the example returns 0.0 when the request fails. To GRPO a dead judge then
                                     looks like "every sample is wrong": zero-variance groups, no gradient, no
                                     error. Here a request is retried (openai client: connection errors, 429,
                                     5xx, with backoff) and then raises, which ends the run.
    the reply cannot be malformed    the judge's Output schema is enforced by vLLM (response_format json_schema),
                                     with temperature and max_tokens set. A reply that still does not parse
                                     (cut by max_tokens) scores 0 and is counted: environment/judge_unparsed.
    one client per server            shared by all env instances; requests go round-robin over the servers.
    a judge per row                  reward_spec.judge of the dataset row names its judge (judges.py), so one run
                                     can mix datasets that are graded differently; judge.name is the default.
    settings come from the config's `judge:` section, as JUDGE_* variables (launch/render.py -> job.env).
    SkyRL's typed env config has fields only for its own example (environment.skyrl_gym.llm_as_a_judge), so
    it is not used.

Concurrency: the generator runs env.step in a thread pool (environment.skyrl_gym.max_env_workers, default 32),
one trajectory per coroutine, so up to that many judge requests are in flight. Only with generator.batched=false:
the batched path awaits the steps one by one (skyrl_gym_generator.py, generate_batched). render.py rejects
batched=true for a judge config.

Logged per step, mean over the trajectories (environment/<name>):
    judge_unparsed      share of replies that did not fit the schema (reward 0). Should be 0. With a thinking
                        judge: it ran out of judge.max_tokens before the verdict.
    judge_skipped       share decided without the judge (empty completion, judge's precheck)
    judge_seconds       request latency
    judge_<field>       every bool / number the judge answered, e.g. judge_correct
    reward_<data_source>            the reward, per dataset (the row's data_source column): the training data is
                        a collection of datasets, and the mean over all of them hides which one moves
    rule_reward, judge_rule_agree   only for judges with a rule (gsm8k): the rule-based reward of the same
                        completions, and how often judge and rule agree on reward > 0
"""
import glob
import itertools
import json
import os
import re
import threading
import time
import urllib.request

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from skyrl_extensions.judge.judges import JUDGES

TRUNCATION_MARKER = "\n[... truncated ...]\n"


def truncate(text, max_chars):
    """text, or if longer than max_chars: its head (2/3) and tail (1/3) with a marker between."""
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(TRUNCATION_MARKER)
    head = keep * 2 // 3
    return text[:head] + TRUNCATION_MARKER + text[len(text) - (keep - head):]


def server_urls(wait_s):
    """
    The judge servers' base URLs (.../v1). judge.base_url (JUDGE_URL, comma-separated) if set; otherwise the
    files the job's judge nodes publish once their servers are healthy ($RUN_DIR/judge_urls/<node>.txt, one
    URL per line), waiting up to wait_s for all JUDGE_NODES of them.
    """
    if os.environ.get("JUDGE_URL"):
        return [u.strip().rstrip("/") for u in os.environ["JUDGE_URL"].split(",") if u.strip()]
    nodes = int(os.environ.get("JUDGE_NODES", "0"))
    if nodes < 1:
        raise RuntimeError("no judge: the config needs a judge: section with nodes >= 1 or a base_url")
    pattern = os.path.join(os.environ["RUN_DIR"], "judge_urls", "*.txt")
    deadline = time.monotonic() + wait_s
    while len(glob.glob(pattern)) < nodes:
        if time.monotonic() > deadline:
            raise RuntimeError("judge servers not ready after {} s ({}/{} nodes published in {}); vLLM logs: {}".format(
                wait_s, len(glob.glob(pattern)), nodes, os.path.dirname(pattern), os.path.join(os.environ["RUN_DIR"], "judge")))
        time.sleep(10)
    return [line.strip().rstrip("/") for path in sorted(glob.glob(pattern)) for line in open(path) if line.strip()]


class JudgeClient:
    def __init__(self, wait_s=0):
        from openai import OpenAI

        env = os.environ
        if env.get("JUDGE_NAME") not in JUDGES:
            raise RuntimeError("judge.name={!r}: not in skyrl_extensions/judge/judges.py ({})".format(
                env.get("JUDGE_NAME"), ", ".join(sorted(JUDGES))))
        self.judge = JUDGES[env["JUDGE_NAME"]]
        self.model = env["JUDGE_SERVED_NAME"]
        self.temperature = float(env.get("JUDGE_TEMPERATURE", "0"))
        self.max_tokens = int(env.get("JUDGE_MAX_TOKENS", "1024"))
        self.max_chars = int(env.get("JUDGE_MAX_CHARS", "60000"))
        self.extra_body = json.loads(env.get("JUDGE_EXTRA_BODY") or "{}")

        self.urls = server_urls(wait_s)
        for url in self.urls:   # fail now, not in the first rollout: wrong address, wrong model name
            with urllib.request.urlopen(url + "/models", timeout=30) as resp:
                served = [m["id"] for m in json.load(resp)["data"]]
            if self.model not in served:
                raise RuntimeError("judge server {} serves {}, the config expects '{}'".format(url, served, self.model))
        self.clients = [OpenAI(base_url=u, api_key=env.get("JUDGE_API_KEY", "EMPTY"),
                               timeout=float(env.get("JUDGE_TIMEOUT_S", "300")), max_retries=int(env.get("JUDGE_RETRIES", "4")))
                        for u in self.urls]
        self._next = itertools.count()
        self._unparsed_shown = 0
        print("judge '{}': model {} on {} server(s): {}".format(env["JUDGE_NAME"], self.model, len(self.urls), " ".join(self.urls)), flush=True)

    def judge_for(self, extras):
        """The row's judge (reward_spec.judge), or the config's judge.name."""
        name = (extras.get("reward_spec") or {}).get("judge")
        if not name:
            return self.judge
        if name not in JUDGES:
            raise RuntimeError("dataset row asks for judge {!r}: not in judges.py ({})".format(name, ", ".join(sorted(JUDGES))))
        return JUDGES[name]

    def ask(self, r, judge):
        """-> (scores dict, or None if the reply does not fit the schema; raw reply). Raises when the server fails."""
        m = judge.messages(r)
        messages = [{"role": "user", "content": m}] if isinstance(m, str) else m
        client = self.clients[next(self._next) % len(self.clients)]
        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": judge.schema(), "strict": True}},
            extra_body=self.extra_body)
        # None when a thinking judge used up max_tokens before its verdict (served with a reasoning parser)
        text = resp.choices[0].message.content or ""
        try:
            return judge.parse(text), text
        except ValueError as e:
            if self._unparsed_shown < 5:   # enough to see why; the rate is in environment/judge_unparsed
                self._unparsed_shown += 1
                print("judge reply does not fit Output (finish_reason={}, reward 0): {} | reply: {!r}".format(
                    resp.choices[0].finish_reason, (str(e).splitlines() or [""])[0], text[:500]), flush=True)
            return None, text


_client = None
_lock = threading.Lock()


def connect(wait_s=0):
    """The process-wide JudgeClient. The entrypoint calls this first with the startup timeout."""
    global _client
    with _lock:
        if _client is None:
            _client = JudgeClient(wait_s)
        return _client


class JudgeEnv(BaseTextEnv):
    def __init__(self, env_config=None, extras=None):
        super().__init__()
        self.extras = extras or {}
        assert "reward_spec" in self.extras and "ground_truth" in self.extras["reward_spec"], \
            "the dataset row needs reward_spec.ground_truth"
        self.prompt = None
        self.metrics = {}

    def init(self, prompt):
        self.prompt = prompt
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        client = connect()
        judge = client.judge_for(self.extras)
        r = {"prompt": self.prompt, "completion": action, "shown": truncate(action, client.max_chars), "extras": self.extras}
        metrics = {"judge_unparsed": 0.0, "judge_skipped": 0.0}

        reward = 0.0 if not action.strip() else judge.precheck(r)
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

        rule = judge.rule(r)
        if rule is not None:
            metrics["rule_reward"] = float(rule)
            metrics["judge_rule_agree"] = float((reward > 0) == (rule > 0))
        source = self.extras.get("data_source")
        if source:
            metrics["reward_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source))] = float(reward)
        self.metrics = metrics
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})

    def get_metrics(self):
        return self.metrics
