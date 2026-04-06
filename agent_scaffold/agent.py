"""
Agent loop for VulnLLM-R scaffold.

Paper description (arXiv:2512.07533):
  For each function, extract all functions along 3 randomly sampled paths from
  the project's entry point to this function in the call graph as initial context.
  Equip the model with a tool to retrieve function implementations by name,
  restricting the number of interaction rounds.

  Two testing strategies (also from paper):
  - Truncated generation: stop reasoning chain at a token limit, force final answer.
    (Handled at the model/vLLM level via max_tokens; no special logic here.)
  - Policy-based generation: query model 4× without CWE hint to collect candidate
    CWE types, then add those as a policy hint for the final authoritative query.
    Implemented in run_agent_with_policy().

Tool use protocol (text-based, works with any model):
  Model may output [RETRIEVE: function_name] before ## Final Answer.
  Scaffold detects this, injects the function body, and re-prompts.
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vulscan.utils.sys_prompts import (
    long_context_reasoning_user_prompt,
    reasoning_user_prompt,
)
from vulscan.utils.get_cwe_info import get_cwe_info

_RETRIEVE_HINT = (
    "\nIf you need to inspect another function's implementation, "
    "output [RETRIEVE: function_name] before your final answer "
    "(up to {max_rounds} retrieval(s) allowed).\n"
)

_REASONING = "You should STRICTLY structure your response as follows:"

_POLICY_PREFIX = "You should only focus on checking and reasoning if the code contains one of the following CWEs:"


def _build_policy_str(cwe_ids: list[str]) -> str:
    """Build the CWE_INFO policy string from a list of CWE IDs."""
    policy = _POLICY_PREFIX
    for cwe_id in cwe_ids:
        try:
            num = int(cwe_id.split("-")[-1])
            desc = get_cwe_info(num)
            if "Unknown CWE" not in desc:
                policy += f"\n- {cwe_id}: {desc}"
        except (ValueError, Exception):
            policy += f"\n- {cwe_id}"
    return policy


def _build_prompt(target_body: str, context_pairs: list[tuple[str, str]],
                  max_rounds: int, cwe_policy: str = "") -> str:
    retrieve_hint = _RETRIEVE_HINT.format(max_rounds=max_rounds) if max_rounds > 0 else ""
    if context_pairs:
        ctx = "// context\n" + "\n\n".join(body for _, body in context_pairs)
        code = f"{ctx}\n\n// target function\n{target_body}"
        return long_context_reasoning_user_prompt.format(
            CODE=code,
            CWE_INFO=cwe_policy,
            REASONING=_REASONING,
            ADDITIONAL_CONSTRAINT=retrieve_hint,
        )
    else:
        return reasoning_user_prompt.format(
            CODE=target_body,
            CWE_INFO=cwe_policy,
            REASONING=_REASONING,
            ADDITIONAL_CONSTRAINT=retrieve_hint,
        )


def _parse_retrieve(text: str) -> str | None:
    m = re.search(r"\[RETRIEVE:\s*(\w+)\s*\]", text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_answer(output: str) -> tuple[str, str]:
    """Parse #judge: yes/no and #type: CWE-XX from model output."""
    text = output.split("## Final Answer")[-1].lower()
    judge, cwe_type = "unknown", "N/A"
    try:
        judge = text.split("#judge: ")[1].split("\n")[0].strip()
    except IndexError:
        pass
    try:
        cwe_type = text.split("#type: ")[1].split("\n")[0].strip().upper()
    except IndexError:
        pass

    if "yes" in judge and "no" not in judge:
        judge = "yes"
    elif "no" in judge and "yes" not in judge:
        judge = "no"
    else:
        judge = "unknown"

    return judge, cwe_type


def run_agent(
    model_fn,
    target_name: str,
    target_body: str,
    context_pairs: list[tuple[str, str]],
    all_functions: dict[str, str],
    max_rounds: int = 2,
    cwe_policy: str = "",
    verbose: bool = False,
) -> tuple[str, str, str, int]:
    """
    Run the agent loop for a single target function.

    Args:
        model_fn: callable(prompt: str) -> str
        target_name: name of the function being analyzed
        target_body: source of the target function
        context_pairs: [(name, body)] for context functions
        all_functions: full project function map for retrieval
        max_rounds: max tool-retrieval rounds
        cwe_policy: CWE_INFO string to inject (empty = no hint)
        verbose: print debug info

    Returns:
        (judge, cwe_type, full_output, rounds_used)
        judge: "yes" | "no" | "unknown"
    """
    prompt = _build_prompt(target_body, context_pairs, max_rounds, cwe_policy)
    conversation = prompt
    rounds_used = 0
    last_output = ""

    for _ in range(max_rounds + 1):  # initial call + up to max_rounds retrievals
        if verbose:
            print(f"    [agent] prompt_len={len(conversation)}")

        output = model_fn(conversation)
        last_output = output or ""

        if not last_output:
            break

        if verbose:
            print(f"    [agent] output: {last_output[:300]!r}")

        # If model already produced final answer, stop
        if "## Final Answer" in last_output:
            break

        # Check for retrieval request
        if rounds_used < max_rounds:
            func_name = _parse_retrieve(last_output)
            if func_name and func_name in all_functions:
                rounds_used += 1
                retrieved = all_functions[func_name]
                if verbose:
                    print(f"    [agent] retrieving '{func_name}' (round {rounds_used})")
                conversation = (
                    conversation + "\n\n" + last_output
                    + f"\n\n[Retrieved: {func_name}]\n```c\n{retrieved}\n```\n"
                    + "Continue your analysis:"
                )
                continue

        break  # no retrieval or max rounds reached

    judge, cwe_type = extract_answer(last_output)
    return judge, cwe_type, last_output, rounds_used

def run_agent_with_policy(
    model_fn,
    target_name: str,
    target_body: str,
    context_pairs: list[tuple[str, str]],
    all_functions: dict[str, str],
    max_rounds: int = 2,
    policy_runs: int = 4,
    model_fn_diverse=None,
    verbose: bool = False,
) -> tuple[str, str, str, int, dict, list]:
    """
    Returns:
        (judge, cwe_type, final_output, rounds_used, policy_cwes, exploratory_results)
        exploratory_results: list of {run, judge, cwe_type, output} for each exploratory run
    """
    if model_fn_diverse is None:
        model_fn_diverse = model_fn

    # Step 1: Exploratory runs — collect CWE candidates
    collected: Counter = Counter()
    exploratory_results = []
    for i in range(policy_runs):
        if verbose:
            print(f"    [policy] exploratory run {i + 1}/{policy_runs}")
        judge, cwe_type, output, rounds = run_agent(
            model_fn_diverse, target_name, target_body, context_pairs,
            all_functions, max_rounds, cwe_policy="", verbose=verbose,
        )
        exploratory_results.append({
            "run": i + 1,
            "judge": judge,
            "cwe_type": cwe_type,
            "rounds_used": rounds,
            "output": output,
        })
        if judge == "yes" and cwe_type and cwe_type != "N/A":
            for cwe in re.findall(r"CWE-\d+", cwe_type.upper()):
                collected[cwe] += 1

    if verbose:
        print(f"    [policy] collected candidates: {dict(collected)}")

    # Step 2: Build policy from collected CWEs (most-voted first)
    candidate_cwes = [cwe for cwe, _ in collected.most_common()]

    # If no CWEs collected, skip final query and return no
    if not candidate_cwes:
        return "no", "N/A", "", 0, dict(collected), exploratory_results
    
    policy_str = _build_policy_str(candidate_cwes) if candidate_cwes else ""

    if verbose:
        if policy_str:
            print(f"    [policy] final query with policy: {candidate_cwes}")
        else:
            print("    [policy] no candidates collected, final query without policy")

    # Step 3: Final authoritative query with policy hint
    judge, cwe_type, output, rounds = run_agent(
        model_fn, target_name, target_body, context_pairs,
        all_functions, max_rounds, cwe_policy=policy_str, verbose=verbose,
    )

    return judge, cwe_type, output, rounds, dict(collected), exploratory_results