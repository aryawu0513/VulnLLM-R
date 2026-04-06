"""
Gradio results viewer for VulnLLM-R evaluation results.
Groups samples by: language → category → variant → attack

Usage: python results_viewer.py
"""
import hashlib
import json
import re
from pathlib import Path
import gradio as gr

BASE      = Path(__file__).parent
REPO_ROOT = BASE.parent

NPD_C_VARIANTS  = ["creatend", "findrec", "mkbuf", "allocate"]
UAF_C_VARIANTS  = ["freeitem", "dropconn", "relogger", "rmentry"]
PYTHON_VARIANTS = ["finduser", "makeconn", "parseitem", "loadconf"]
CATEGORIES      = ["clean", "dpi", "dpi_think", "context_aware",
                   "annotated_clean", "annotated_dpi", "annotated_context_aware"]

BUG_TYPE = "NPD"
DEFENSE  = None
KNOWN_DEFENSES = ["D1", "D2", "D3L", "D3A", "D3B", "D4_append", "D4_prepend"]

# ── Judge cache ───────────────────────────────────────────────────────────────

def load_judge_cache() -> dict:
    p = REPO_ROOT / "judge" / "judge_cache.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

JUDGE_CACHE = load_judge_cache()


def get_judge(bug_type: str, lang: str, category: str, variant: str, attack: str) -> dict:
    lang_key = "python" if lang == "Python" else "c"
    key = f"vulnllm:{bug_type}:{lang_key}:{category}:{variant}:{attack}"
    return JUDGE_CACHE.get(key, {})


# ── Index building ─────────────────────────────────────────────────────────────
# Index key: (category, variant, attack)
# Value: {code, flag, output, predicted_is_vulnerable, predicted_vulnerability_type,
#          completion_tokens}
# Built once per (lang, bug_type, defense); queries never require a rebuild.

def build_index(lang: str) -> dict:
    """
    Scan all CATEGORIES × variants from dataset files (canonical source of truth).
    Join result records by full-code MD5 hash.
    Returns {(category, variant, attack) → record_dict}.
    """
    lang_dir  = "Python" if lang == "Python" else "C"
    ds_subdir = "python" if lang == "Python" else "c"
    variants  = PYTHON_VARIANTS if lang == "Python" else (
                UAF_C_VARIANTS if BUG_TYPE == "UAF" else NPD_C_VARIANTS)

    # D3L injects [VERIFIABLE]/[INTENDED]/[ADVERSARIAL] labels into code comments at
    # runtime; strip them before hashing so we can match against unmodified datasets.
    _D3L_LABEL_RE = re.compile(r'\[(VERIFIABLE|INTENDED|ADVERSARIAL)\]\s*')

    def _code_hashes(code: str):
        """Return the set of MD5 hashes to index this code under."""
        h_raw = hashlib.md5(code.encode()).hexdigest()
        stripped = _D3L_LABEL_RE.sub('', code)
        h_stripped = hashlib.md5(stripped.encode()).hexdigest()
        return {h_raw, h_stripped}

    # Step 1: collect result records across all categories, indexed by full code hash
    hash_to_result: dict = {}
    for cat in CATEGORIES:
        if DEFENSE and DEFENSE != "baseline":
            result_dir = BASE / f"results-defense/{DEFENSE}/{lang_dir}/{BUG_TYPE}/policy/{cat}"
        else:
            result_dir = BASE / f"results/{lang_dir}/{BUG_TYPE}/policy/{cat}"
        if not result_dir.exists():
            continue
        for rf in sorted(result_dir.glob("*.json")):
            try:
                records = json.loads(rf.read_text())
            except Exception:
                continue
            for rec in records:
                if "input" not in rec:
                    continue
                inp = rec["input"]
                cs = inp.find("```")
                if cs == -1:
                    continue
                cs += 3
                ce = inp.find("```", cs)
                code = inp[cs:ce].strip() if ce > cs else ""
                for h in _code_hashes(code):
                    hash_to_result[h] = rec

    # Step 2: walk dataset files for each category/variant to enumerate all attacks
    index: dict = {}
    for cat in CATEGORIES:
        for variant in variants:
            if DEFENSE and DEFENSE != "baseline" and (BASE / f"datasets-defense/{DEFENSE}").exists():
                ds_dir = BASE / f"datasets-defense/{DEFENSE}/{lang_dir}/{BUG_TYPE}/{cat}/{variant}/{ds_subdir}"
                if not ds_dir.exists():
                    ds_dir = BASE / f"datasets/{lang_dir}/{BUG_TYPE}/{cat}/{variant}/{ds_subdir}"
            else:
                ds_dir = BASE / f"datasets/{lang_dir}/{BUG_TYPE}/{cat}/{variant}/{ds_subdir}"
            if not ds_dir.exists():
                continue
            for dsf in sorted(ds_dir.glob("*.json")):
                try:
                    ds = json.loads(dsf.read_text())
                    code = ds[0]["code"] if isinstance(ds, list) and ds else ""
                except Exception:
                    continue
                attack = dsf.stem
                if attack.startswith(variant + "_"):
                    attack = attack[len(variant) + 1:]
                rec = {}
                for h in _code_hashes(code.strip()):
                    if h in hash_to_result:
                        rec = hash_to_result[h]
                        break
                index[(cat, variant, attack)] = {
                    "code":                         code,
                    "flag":                         rec.get("flag"),
                    "output":                       rec.get("output", ""),
                    "predicted_is_vulnerable":      rec.get("predicted_is_vulnerable", "?"),
                    "predicted_vulnerability_type": rec.get("predicted_vulnerability_type", "?"),
                    "completion_tokens":            rec.get("completion_tokens", "?"),
                    "judge":                        get_judge(BUG_TYPE, lang, cat, variant, attack),
                    "category":                     cat,
                }
    return index


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_incompetent(flag, judge: dict, category: str) -> bool:
    if flag != "fn":
        return False
    return (judge and judge.get("is_incompetent")) or category == "clean"


def flag_badge_html(flag, judge: dict = None, category: str = "") -> str:
    if flag is None:
        return ('<span style="background:#e2e3e5;color:#383d41;padding:6px 16px;'
                'border-radius:4px;font-weight:bold;font-size:1.1em">— NO RESULT —</span>')
    if _is_incompetent(flag, judge or {}, category):
        return ('<span style="background:#fff3cd;color:#856404;padding:6px 16px;'
                'border-radius:4px;font-weight:bold;font-size:1.1em">⚠ INCOMPETENT</span>')
    styles = {
        "tp": ("✓ DETECTED",  "#d4edda", "#155724"),
        "fn": ("✗ EVADED",    "#f8d7da", "#721c24"),
        "fp": ("⚠ FALSE POS", "#fff3cd", "#856404"),
        "tn": ("✓ CLEAN",     "#d4edda", "#155724"),
    }
    label, bg, fg = styles.get(flag, (f"? {str(flag).upper()}", "#e2e3e5", "#383d41"))
    return (f'<span style="background:{bg};color:{fg};padding:6px 16px;'
            f'border-radius:4px;font-weight:bold;font-size:1.1em">{label}</span>')


def flag_badge(flag, judge: dict = None, category: str = "") -> str:
    if flag is None:
        return "*(no result)*"
    if _is_incompetent(flag, judge or {}, category):
        return "**INCOMPETENT** — system limitation (not attack-induced)"
    return {
        "tp": "**TP** — detected (true positive)",
        "fn": "**FN** — missed (attack succeeded)",
        "fp": "**FP** — false alarm (false positive)",
        "tn": "**TN** — correctly clean (true negative)",
    }.get(flag, f"**{flag}**")


# ── Global state ──────────────────────────────────────────────────────────────

INDEX        = {}
CURRENT_LANG = "C"


def _rebuild(lang: str, bug_type: str = None, defense: str = None):
    global INDEX, CURRENT_LANG, BUG_TYPE, DEFENSE, JUDGE_CACHE
    CURRENT_LANG = lang
    if bug_type is not None:
        BUG_TYPE = bug_type
    if defense is not None:
        DEFENSE = None if defense == "baseline" else defense
    JUDGE_CACHE = load_judge_cache()
    INDEX = build_index(lang)


# ── UI helpers ────────────────────────────────────────────────────────────────

def get_variants(category: str):
    return sorted(set(v for (c, v, _) in INDEX if c == category)) or []


def get_attacks(category: str, variant: str):
    return sorted(a for (c, v, a) in INDEX if c == category and v == variant)


# ── Display ───────────────────────────────────────────────────────────────────

def _display(category: str, variant, attack):
    key = (category, variant, attack)
    if not category or not variant or not attack or key not in INDEX:
        return "(no data)", "", "", ""
    rec       = INDEX[key]
    flag      = rec.get("flag")
    judge     = rec.get("judge", {})
    category  = rec.get("category", category)
    code      = rec.get("code", "")
    model_out = rec.get("output", "")
    badge     = flag_badge_html(flag, judge, category)
    judgment  = "\n".join([
        flag_badge(flag, judge, category),
        "",
        f"**Predicted vulnerable**: `{rec.get('predicted_is_vulnerable', '?')}`",
        f"**Predicted type**: `{rec.get('predicted_vulnerability_type', '?')}`",
        f"**Completion tokens**: {rec.get('completion_tokens', '?')}",
    ])
    return code, model_out, badge, judgment


# ── Callbacks ─────────────────────────────────────────────────────────────────

def on_defense_change(defense: str, lang: str, category: str, variant: str, attack: str):
    _rebuild(lang, defense=defense)
    code, model_out, badge, judgment = _display(category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return gr.update(value=code, language=code_lang), model_out, badge, judgment


def on_bug_type_change(bug_type: str, lang: str, category: str):
    actual_lang = "C" if bug_type == "UAF" else lang
    _rebuild(actual_lang, bug_type=bug_type)
    variants = get_variants(category)
    variant  = variants[0] if variants else None
    attacks  = get_attacks(category, variant) if variant else []
    attack   = attacks[0] if attacks else None
    code, model_out, badge, judgment = _display(category, variant, attack)
    code_lang = "python" if actual_lang == "Python" else "c"
    return (
        gr.update(value=actual_lang, interactive=(bug_type == "NPD")),
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        model_out, badge, judgment,
    )


def on_lang_change(lang: str, category: str):
    _rebuild(lang)
    variants = get_variants(category)
    variant  = variants[0] if variants else None
    attacks  = get_attacks(category, variant) if variant else []
    attack   = attacks[0] if attacks else None
    code, model_out, badge, judgment = _display(category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        model_out, badge, judgment,
    )


def on_category_change(lang: str, category: str):
    variants = get_variants(category)
    variant  = variants[0] if variants else None
    attacks  = get_attacks(category, variant) if variant else []
    attack   = attacks[0] if attacks else None
    code, model_out, badge, judgment = _display(category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        model_out, badge, judgment,
    )


def on_variant_change(lang: str, category: str, variant: str):
    attacks = get_attacks(category, variant)
    attack  = attacks[0] if attacks else None
    code, model_out, badge, judgment = _display(category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=attacks, value=attack),
        gr.update(value=code, language=code_lang),
        model_out, badge, judgment,
    )


def on_attack_change(lang: str, category: str, variant: str, attack: str):
    code, model_out, badge, judgment = _display(category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return gr.update(value=code, language=code_lang), model_out, badge, judgment


# ── Initial state ─────────────────────────────────────────────────────────────

_rebuild("C")

init_cat      = "clean"
init_variants = get_variants(init_cat)
init_variant  = init_variants[0] if init_variants else None
init_attacks  = get_attacks(init_cat, init_variant) if init_variant else []
init_attack   = init_attacks[0] if init_attacks else None
init_code, init_out, init_badge, init_judgment = _display(init_cat, init_variant, init_attack)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="VulnLLM-R Results Viewer") as demo:
    gr.Markdown("# VulnLLM-R Results Viewer")
    gr.Markdown("Browse per-attack model outputs and detection outcomes.")

    with gr.Row():
        bug_type_dd = gr.Radio(choices=["NPD", "UAF"], value="NPD", label="Bug Type")
        lang_radio  = gr.Radio(choices=["C", "Python"], value="C", label="Language")
        defense_dd  = gr.Dropdown(choices=["baseline"] + KNOWN_DEFENSES, value="baseline",
                                  label="Defense")
        cat_dd      = gr.Dropdown(choices=CATEGORIES,    value=init_cat,      label="Category")
        var_dd      = gr.Dropdown(choices=init_variants, value=init_variant,  label="Variant")
        atk_dd      = gr.Dropdown(choices=init_attacks,  value=init_attack,   label="Attack")

    verdict_html = gr.HTML(value=init_badge, label="")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Code shown to model")
            code_box = gr.Code(language="c", label="", lines=30, value=init_code)
        with gr.Column():
            gr.Markdown("### Model output")
            out_box = gr.Textbox(label="", lines=30, max_lines=80,
                                 show_copy_button=True, value=init_out)

    judgment_md = gr.Markdown(value=init_judgment)

    # Wire events
    defense_dd.change(
        on_defense_change,
        inputs=[defense_dd, lang_radio, cat_dd, var_dd, atk_dd],
        outputs=[code_box, out_box, verdict_html, judgment_md],
    )
    bug_type_dd.change(
        on_bug_type_change,
        inputs=[bug_type_dd, lang_radio, cat_dd],
        outputs=[lang_radio, var_dd, atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    lang_radio.change(
        on_lang_change,
        inputs=[lang_radio, cat_dd],
        outputs=[var_dd, atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    cat_dd.change(
        on_category_change,
        inputs=[lang_radio, cat_dd],
        outputs=[var_dd, atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    var_dd.change(
        on_variant_change,
        inputs=[lang_radio, cat_dd, var_dd],
        outputs=[atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    atk_dd.change(
        on_attack_change,
        inputs=[lang_radio, cat_dd, var_dd, atk_dd],
        outputs=[code_box, out_box, verdict_html, judgment_md],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=6002)
