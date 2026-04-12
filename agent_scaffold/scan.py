"""
VulnLLM-R agent scaffold scanner.

Implements the paper's agent scaffold (arXiv:2512.07533):
  - Parse all functions in a project
  - Build the call graph
  - For each function: sample 3 paths from entry point, collect context
  - Run the model with optional tool-based context retrieval
  - Report findings

Usage:
  # vLLM (VulnLLM-R-7B, needs GPU):
  python -m agent_scaffold.scan --repo agent_scaffold/demo_repo/clean \\
      --language c --vllm UCSB-SURFI/VulnLLM-R-7B

  # Anthropic API (claude, for testing without GPU):
  python -m agent_scaffold.scan --repo agent_scaffold/demo_repo/clean \\
      --language c --anthropic claude-sonnet-4-6

  # Run all demo variants and compare:
  python -m agent_scaffold.scan --demo --anthropic claude-sonnet-4-6
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_scaffold.call_graph import (
    parse_project,
    build_call_graph,
    find_entry_points,
    get_context_functions,
)
from agent_scaffold.agent import run_agent, run_agent_with_policy, _build_policy_str


def _build_hint_policy(cwe_hints: list[str]) -> str:
    return _build_policy_str(cwe_hints) if cwe_hints else ""


def _extract_project_audit(repo_dir: str, language: str) -> str:
    """Extract [Comment Audit]...[End Audit] blocks from source files.

    Mirrors RepoAudit's _extract_audit_block() in intra_dataflow_analyzer.py:
    reads full source files (not tree-sitter-extracted bodies) so the D4_prepend
    audit block placed at the file top is visible.
    """
    ext_map = {"c": [".c"], "cpp": [".c", ".cpp", ".h"], "python": [".py"]}
    extensions = set(ext_map.get(language, [".c"]))
    blocks = []
    for path in sorted(Path(repo_dir).rglob("*")):
        if path.suffix in extensions and path.is_file():
            try:
                content = path.read_text(errors="replace")
                m = re.search(r'\[Comment Audit\](.*?)\[End Audit\]', content, re.DOTALL)
                if m:
                    blocks.append('[Comment Audit]' + m.group(1) + '[End Audit]')
            except Exception:
                pass
    return "\n\n".join(blocks) if blocks else ""


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------
def make_vllm_fns(model_name: str, max_tokens: int = 4096):
    from vllm import SamplingParams
    from model_zoo.vllm_model import VllmModel
    from vulscan.utils.sys_prompts import qwen_sys_prompt

    model = VllmModel(
        model=model_name,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        num_gpus=1,
        seed=None,  # allow diverse sampling across exploratory runs
    )
    system_prompt = qwen_sys_prompt

    def make_fn(temperature: float):
        def model_fn(prompt: str) -> str:
            outputs, _, _, _ = model.run(
                eval_examples=[{"input": prompt, "output": ""}],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return outputs[0][0]  # outputs is list of lists
        return model_fn

    return make_fn(0.0), make_fn(0.6)


def make_anthropic_fn(model_name: str, max_tokens: int = 4096, temperature: float = 0.0):
    import anthropic
    client = anthropic.Anthropic()

    def model_fn(prompt: str) -> str:
        msg = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    return model_fn


def make_litellm_fn(model_name: str, max_tokens: int = 4096, temperature: float = 0.0):
    import litellm

    def model_fn(prompt: str) -> str:
        resp = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    return model_fn


def make_model_fns(args):
    """Return (model_fn, model_fn_diverse) pair based on CLI args."""
    if args.vllm:
        return make_vllm_fns(args.vllm, max_tokens=args.max_tokens)

    maker = {"anthropic": make_anthropic_fn, "api": make_litellm_fn}
    backend = "anthropic" if args.anthropic else "api"
    model_name = args.anthropic or args.api
    make = maker[backend]
    model_fn = make(model_name, temperature=0.0, max_tokens=args.max_tokens)
    model_fn_diverse = make(model_name, temperature=0.6, max_tokens=args.max_tokens) if args.policy_runs > 0 else model_fn
    return model_fn, model_fn_diverse


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_project(
    repo_dir: str,
    language: str,
    model_fn,
    n_paths: int = 3,
    max_rounds: int = 2,
    policy_runs: int = 0,
    model_fn_diverse=None,
    target_functions: list[str] | None = None,
    cwe_hints: list[str] | None = None,
    verbose: bool = False,
) -> list[dict]:
    print(f"\n[scan] repo: {repo_dir}")
    if policy_runs > 0:
        print(f"[scan] mode: policy-based ({policy_runs} exploratory runs + 1 final)")
    all_functions = parse_project(repo_dir, language)
    print(f"[scan] functions found: {list(all_functions.keys())}")

    # D4_prepend defense: audit block sits at file top (before #includes),
    # invisible to tree-sitter extraction. Read it from the full source file
    # and prepend to target_body, mirroring RepoAudit's _extract_audit_block().
    audit_block = _extract_project_audit(repo_dir, language)
    if audit_block:
        print(f"[scan] audit block found ({len(audit_block)} chars) — prepending to target body")

    call_graph = build_call_graph(all_functions, language)
    entry_points = find_entry_points(call_graph)
    entry = entry_points[0]
    print(f"[scan] entry point: {entry}")

    targets = target_functions or [f for f in all_functions if f != entry]
    results = []

    for target_name in targets:
        print(f"\n  → scanning: {target_name}")
        context = get_context_functions(call_graph, all_functions, entry, target_name, n_paths)
        print(f"     context: {[n for n, _ in context]}")

        target_body = all_functions[target_name]
        if audit_block:
            target_body = f"/*\n{audit_block}\n*/\n\n{target_body}"

        if policy_runs > 0:
            judge, cwe_type, full_output, rounds, policy_cwes, exploratory_results, final_results = run_agent_with_policy(
                model_fn=model_fn,
                target_name=target_name,
                target_body=target_body,
                context_pairs=context,
                all_functions=all_functions,
                max_rounds=max_rounds,
                policy_runs=policy_runs,
                model_fn_diverse=model_fn_diverse,
                default_cwes=cwe_hints,
                verbose=verbose,
            )
            print(f"     policy candidates: {policy_cwes}")
        else:
            cwe_policy = _build_hint_policy(cwe_hints) if cwe_hints else ""
            judge, cwe_type, full_output, rounds = run_agent(
                model_fn=model_fn,
                target_name=target_name,
                target_body=target_body,
                context_pairs=context,
                all_functions=all_functions,
                max_rounds=max_rounds,
                cwe_policy=cwe_policy,
                verbose=verbose,
            )
            policy_cwes = {}
            exploratory_results = []
            final_results = []

        result = {
            "function": target_name,
            "judge": judge,
            "cwe_type": cwe_type,
            "rounds_used": rounds,
            "context_functions": [n for n, _ in context],
            "policy_cwes": policy_cwes,
            "exploratory_results": exploratory_results,
            "final_results": final_results,
            "output": full_output,
        }
        results.append(result)
        flag = "⚠️ VULN" if judge == "yes" else ("✅ SAFE" if judge == "no" else "❓ ?")
        print(f"     {flag}  [{cwe_type}]  (retrieval rounds: {rounds})")

    return results


def print_summary(results: list[dict], label: str = ""):
    vuln = [r for r in results if r["judge"] == "yes"]
    print(f"\n{'='*60}")
    if label:
        print(f"VARIANT: {label}")
    print(f"Scanned: {len(results)}  Vulnerabilities detected: {len(vuln)}")
    for r in results:
        flag = "⚠️ VULN" if r["judge"] == "yes" else ("✅ SAFE" if r["judge"] == "no" else "❓ ?")
        print(f"  {flag}  {r['function']}  [{r['cwe_type']}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_VARIANTS = ["clean", "attack_A1", "attack_A2", "attack_A3", "attack_A4", "attack_A5"]


def main():
    ap = argparse.ArgumentParser(description="VulnLLM-R agent scaffold scanner")
    ap.add_argument("--repo", help="Path to C project directory to scan")
    ap.add_argument("--language", default="c", choices=["c", "cpp", "python", "java"])
    ap.add_argument("--n-paths", type=int, default=3, help="Call-graph paths to sample per function")
    ap.add_argument("--max-rounds", type=int, default=2, help="Max context-retrieval rounds")
    ap.add_argument(
        "--policy-runs", type=int, default=0,
        help="Policy-based generation: N exploratory queries before final (paper uses 4). 0=disabled.",
    )
    ap.add_argument("--target", nargs="+", help="Specific functions to scan (default: all)")
    ap.add_argument("--cwe-hint", nargs="+", metavar="CWE", dest="cwe_hints",
                    help="Pre-seed the policy with these CWEs (e.g. --cwe-hint CWE-476). "
                         "Merged with CWEs collected from exploratory runs.")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--output", help="Save JSON results to file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Run all demo_repo variants and print comparison table",
    )

    mg = ap.add_mutually_exclusive_group(required=True)
    mg.add_argument("--vllm", metavar="MODEL", help="Use vLLM backend (e.g. UCSB-SURFI/VulnLLM-R-7B)")
    mg.add_argument("--anthropic", metavar="MODEL", help="Use Anthropic API (e.g. claude-sonnet-4-6)")
    mg.add_argument("--api", metavar="MODEL", help="Use LiteLLM API (e.g. gpt-4o)")

    args = ap.parse_args()
    model_fn, model_fn_diverse = make_model_fns(args)

    demo_base = Path(__file__).parent / "demo_repo"

    if args.demo:
        all_results = {}
        for variant in DEMO_VARIANTS:
            repo = demo_base / variant
            if not repo.exists():
                print(f"[skip] {variant} (not found)")
                continue
            results = scan_project(
                str(repo), args.language, model_fn,
                args.n_paths, args.max_rounds, args.policy_runs,
                model_fn_diverse, args.target, args.cwe_hints, args.verbose,
            )
            all_results[variant] = results
            print_summary(results, label=variant)

        # Comparison table
        print(f"\n{'='*60}")
        print("ATTACK SUMMARY")
        print(f"{'='*60}")
        print(f"{'Variant':<15} {'fill_buffer':<12} {'CWE':<12} {'Attack succeeded?'}")
        print("-" * 60)
        for variant, results in all_results.items():
            for r in results:
                if r["function"] == "fill_buffer":
                    detected = r["judge"] == "yes"
                    status = "✅ detected" if detected else ("🎯 EVADED" if variant != "clean" else "❌ missed baseline")
                    print(f"{variant:<15} {r['judge']:<12} {r['cwe_type']:<12} {status}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        if not args.repo:
            ap.error("--repo is required unless --demo is used")
        results = scan_project(
            args.repo, args.language, model_fn,
            args.n_paths, args.max_rounds, args.policy_runs,
            model_fn_diverse, args.target, args.cwe_hints, args.verbose,
        )
        print_summary(results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()