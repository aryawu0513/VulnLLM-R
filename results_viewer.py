"""
Gradio results viewer for VulnLLM-R evaluation results.
Groups samples by: language → category → variant → attack

Features:
  - Manual labeling: reclassify EVADED → INCOMPETENT (or back), saved to judge_cache.json
  - Phrase highlighting (red = fooled, green = recognized attack) in model output,
    saved to judge/highlight_cache.json

Usage: python results_viewer.py
"""
import hashlib
import html as _html
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

HIGHLIGHT_CACHE_PATH = REPO_ROOT / "judge" / "highlight_cache.json"

# ── Judge cache ───────────────────────────────────────────────────────────────

def load_judge_cache() -> dict:
    p = REPO_ROOT / "judge" / "judge_cache.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def save_judge_cache(cache: dict):
    p = REPO_ROOT / "judge" / "judge_cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2))


def load_highlight_cache() -> dict:
    if HIGHLIGHT_CACHE_PATH.exists():
        try:
            return json.loads(HIGHLIGHT_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def save_highlight_cache(cache: dict):
    HIGHLIGHT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HIGHLIGHT_CACHE_PATH.write_text(json.dumps(cache, indent=2))


JUDGE_CACHE = load_judge_cache()


def get_judge(bug_type: str, lang: str, category: str, variant: str, attack: str) -> dict:
    lang_key = "python" if lang == "Python" else "c"
    key = f"vulnllm:{bug_type}:{lang_key}:{category}:{variant}:{attack}"
    return JUDGE_CACHE.get(key, {})


def _hl_cache_key(lang: str, bug_type: str, category: str, variant: str, attack: str) -> str:
    lang_key = "python" if lang == "Python" else "c"
    return f"vulnllm:{lang_key}:{bug_type}:{category}:{variant}:{attack}"


# ── Highlight helpers ─────────────────────────────────────────────────────────

_HL_STYLES = {
    "red":   "background:#fde8e8;color:#c0392b;font-weight:600;border-radius:2px;padding:0 2px",
    "green": "background:#d5f5e3;color:#1e8449;font-weight:600;border-radius:2px;padding:0 2px",
}


def _normalize_highlights(raw: list) -> list[dict]:
    """Migrate old string lists to [{phrase, color}] dicts."""
    result = []
    for h in raw:
        if isinstance(h, str):
            result.append({"phrase": h, "color": "red"})
        elif isinstance(h, dict) and "phrase" in h:
            result.append(h)
    return result


def _render_highlighted(text: str, highlights: list) -> str:
    escaped    = _html.escape(text)
    normalized = _normalize_highlights(highlights)
    for h in sorted(normalized, key=lambda x: len(x.get("phrase", "")), reverse=True):
        phrase = h.get("phrase", "").strip()
        if not phrase:
            continue
        style = _HL_STYLES.get(h.get("color", "red"), _HL_STYLES["red"])
        ep    = _html.escape(phrase)
        escaped = escaped.replace(ep, f'<mark style="{style}">{ep}</mark>')
    return (
        '<pre style="white-space:pre-wrap;font-family:\'Courier New\',monospace;'
        'font-size:0.82em;line-height:1.65;padding:14px;background:#f8f9fa;'
        'border-radius:6px;max-height:580px;overflow-y:auto;border:1px solid #dee2e6">'
        + escaped + '</pre>'
    )


def _highlights_md(highlights: list) -> str:
    normalized = _normalize_highlights(highlights)
    if not normalized:
        return "*No highlights saved*"
    icons = {"red": "🔴", "green": "🟢"}
    items = " &nbsp;·&nbsp; ".join(
        f"{icons.get(h.get('color','red'), '🔴')} `{h.get('phrase','')}`"
        for h in normalized
    )
    return f"**Highlighted:** {items}"


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(lang: str) -> dict:
    lang_dir  = "Python" if lang == "Python" else "C"
    ds_subdir = "python" if lang == "Python" else "c"
    variants  = PYTHON_VARIANTS if lang == "Python" else (
                UAF_C_VARIANTS if BUG_TYPE == "UAF" else NPD_C_VARIANTS)

    _D3L_LABEL_RE = re.compile(r'\[(VERIFIABLE|INTENDED|ADVERSARIAL)\]\s*')

    def _code_hashes(code: str):
        h_raw = hashlib.md5(code.encode()).hexdigest()
        stripped = _D3L_LABEL_RE.sub('', code)
        h_stripped = hashlib.md5(stripped.encode()).hexdigest()
        return {h_raw, h_stripped}

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


def _label_state(lang: str, category: str, variant: str, attack: str):
    key = (category, variant, attack)
    if key not in INDEX:
        return "EVADED", False
    rec  = INDEX[key]
    flag = rec.get("flag")
    cat  = rec.get("category", category)
    if flag != "fn" or cat in ("clean", "annotated_clean"):
        return "EVADED", False
    judge = rec.get("judge", {})
    return ("INCOMPETENT" if judge.get("is_incompetent") else "EVADED"), True


# ── Display ───────────────────────────────────────────────────────────────────

def _display(category: str, variant, attack):
    key = (category, variant, attack)
    if not category or not variant or not attack or key not in INDEX:
        return "(no data)", "", "", "", ""
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


def _display_with_hl(lang: str, category: str, variant: str, attack: str):
    """Returns (code, out_html, raw_out, badge, judgment, lv, lvis, hl_json, hl_md)."""
    result = _display(category, variant, attack)
    if result[0] == "(no data)":
        return "(no data)", _render_highlighted("", []), "", "", "", "EVADED", False, "[]", "*No highlights saved*"
    code, model_out, badge, judgment = result
    hl_cache = load_highlight_cache()
    hl_key   = _hl_cache_key(lang, BUG_TYPE, category, variant, attack)
    hl       = _normalize_highlights(hl_cache.get(hl_key, []))
    out_html = _render_highlighted(model_out, hl)
    hl_md    = _highlights_md(hl)
    lv, lvis = _label_state(lang, category, variant, attack)
    return code, out_html, model_out, badge, judgment, lv, lvis, json.dumps(hl), hl_md


# ── Callbacks ─────────────────────────────────────────────────────────────────

def on_defense_change(defense: str, lang: str, category: str, variant: str, attack: str):
    _rebuild(lang, defense=defense)
    code, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(lang, category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (gr.update(value=code, language=code_lang), out_html, raw, badge, judgment,
            gr.update(value=lv, visible=lvis), gr.update(visible=lvis), gr.update(value=""),
            hl_json, hl_md)


def on_bug_type_change(bug_type: str, lang: str, category: str):
    actual_lang = "C" if bug_type == "UAF" else lang
    _rebuild(actual_lang, bug_type=bug_type)
    variants = get_variants(category)
    variant  = variants[0] if variants else None
    attacks  = get_attacks(category, variant) if variant else []
    attack   = attacks[0] if attacks else None
    code, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(actual_lang, category, variant, attack)
    code_lang = "python" if actual_lang == "Python" else "c"
    return (
        gr.update(value=actual_lang, interactive=(bug_type == "NPD")),
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        out_html, raw, badge, judgment,
        gr.update(value=lv, visible=lvis), gr.update(visible=lvis), gr.update(value=""),
        hl_json, hl_md,
    )


def on_lang_change(lang: str, category: str):
    _rebuild(lang)
    variants = get_variants(category)
    variant  = variants[0] if variants else None
    attacks  = get_attacks(category, variant) if variant else []
    attack   = attacks[0] if attacks else None
    code, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(lang, category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        out_html, raw, badge, judgment,
        gr.update(value=lv, visible=lvis), gr.update(visible=lvis), gr.update(value=""),
        hl_json, hl_md,
    )


def on_category_change(lang: str, category: str):
    variants = get_variants(category)
    variant  = variants[0] if variants else None
    attacks  = get_attacks(category, variant) if variant else []
    attack   = attacks[0] if attacks else None
    code, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(lang, category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        out_html, raw, badge, judgment,
        gr.update(value=lv, visible=lvis), gr.update(visible=lvis), gr.update(value=""),
        hl_json, hl_md,
    )


def on_variant_change(lang: str, category: str, variant: str):
    attacks = get_attacks(category, variant)
    attack  = attacks[0] if attacks else None
    code, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(lang, category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=attacks, value=attack),
        gr.update(value=code, language=code_lang),
        out_html, raw, badge, judgment,
        gr.update(value=lv, visible=lvis), gr.update(visible=lvis), gr.update(value=""),
        hl_json, hl_md,
    )


def on_attack_change(lang: str, category: str, variant: str, attack: str):
    code, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(lang, category, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (gr.update(value=code, language=code_lang), out_html, raw, badge, judgment,
            gr.update(value=lv, visible=lvis), gr.update(visible=lvis), gr.update(value=""),
            hl_json, hl_md)


def on_save_label(lang: str, category: str, variant: str, attack: str, label_val: str):
    global JUDGE_CACHE, INDEX
    lang_key  = "python" if lang == "Python" else "c"
    cache_key = f"vulnllm:{BUG_TYPE}:{lang_key}:{category}:{variant}:{attack}"

    if label_val == "INCOMPETENT":
        JUDGE_CACHE[cache_key] = {"is_incompetent": True}
    else:
        JUDGE_CACHE.pop(cache_key, None)
    save_judge_cache(JUDGE_CACHE)

    INDEX = build_index(lang)
    _, out_html, raw, badge, judgment, lv, lvis, hl_json, hl_md = _display_with_hl(lang, category, variant, attack)
    status = f"✓ Saved **{cache_key}** → **{label_val}**"
    return (badge, judgment,
            gr.update(value=lv, visible=lvis), gr.update(visible=lvis),
            gr.update(value=status))


def on_add_highlight(lang: str, category: str, variant: str, attack: str,
                     phrase: str, color: str, hl_json: str, raw: str):
    hl        = _normalize_highlights(json.loads(hl_json) if hl_json else [])
    phrase    = phrase.strip()
    color_key = (color or "Red").lower()
    if phrase:
        hl = [h for h in hl if h.get("phrase") != phrase]
        hl.append({"phrase": phrase, "color": color_key})
        cache = load_highlight_cache()
        cache[_hl_cache_key(lang, BUG_TYPE, category, variant, attack)] = hl
        save_highlight_cache(cache)
    return _render_highlighted(raw, hl), json.dumps(hl), _highlights_md(hl), ""


def on_clear_highlights(lang: str, category: str, variant: str, attack: str, raw: str):
    cache = load_highlight_cache()
    cache.pop(_hl_cache_key(lang, BUG_TYPE, category, variant, attack), None)
    save_highlight_cache(cache)
    return _render_highlighted(raw, []), json.dumps([]), "*No highlights saved*"


# ── Initial state ─────────────────────────────────────────────────────────────

_rebuild("C")

init_cat      = "clean"
init_variants = get_variants(init_cat)
init_variant  = init_variants[0] if init_variants else None
init_attacks  = get_attacks(init_cat, init_variant) if init_variant else []
init_attack   = init_attacks[0] if init_attacks else None

(init_code, init_out_html, init_raw, init_badge, init_judgment,
 init_lv, init_lvis, init_hl_json, init_hl_md) = _display_with_hl(
    CURRENT_LANG, init_cat, init_variant, init_attack)


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
        label_radio = gr.Radio(
            choices=["EVADED", "INCOMPETENT"],
            label="Manual Classification (override judge cache)",
            value=init_lv,
            visible=init_lvis,
        )
        label_btn = gr.Button("Save Label", size="sm", visible=init_lvis)
    label_status = gr.Markdown("")

    # Persistent state
    hl_state  = gr.State(init_hl_json)
    raw_state = gr.State(init_raw)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Code shown to model")
            code_box = gr.Code(language="c", label="", lines=30, value=init_code)
        with gr.Column():
            gr.Markdown("### Model output")
            out_html_box = gr.HTML(value=init_out_html)
            with gr.Row():
                hl_input     = gr.Textbox(label="Phrase to highlight",
                                          placeholder="Type exact phrase then press Enter or click Add…",
                                          scale=4)
                color_radio  = gr.Radio(["Red", "Green"], value="Red", label="Color", scale=0)
                add_hl_btn   = gr.Button("Add Highlight", size="sm", scale=1)
                clear_hl_btn = gr.Button("Clear All",     size="sm", variant="stop", scale=0)
            hl_display = gr.Markdown(value=init_hl_md)

    judgment_md = gr.Markdown(value=init_judgment)

    # ── Shared output lists ────────────────────────────────────────────────────
    _label_outs = [label_radio, label_btn, label_status]
    _hl_outs    = [out_html_box, hl_state, hl_display, hl_input]
    # Per-navigation outputs (excl. label_status which gets cleared):
    # code_box, out_html_box, raw_state, verdict_html, judgment_md,
    # label_radio, label_btn, label_status, hl_state, hl_display
    _nav_outs = [code_box, out_html_box, raw_state, verdict_html, judgment_md,
                 label_radio, label_btn, label_status, hl_state, hl_display]

    defense_dd.change(
        on_defense_change,
        inputs=[defense_dd, lang_radio, cat_dd, var_dd, atk_dd],
        outputs=_nav_outs,
    )
    bug_type_dd.change(
        on_bug_type_change,
        inputs=[bug_type_dd, lang_radio, cat_dd],
        outputs=[lang_radio, var_dd, atk_dd] + _nav_outs,
    )
    lang_radio.change(
        on_lang_change,
        inputs=[lang_radio, cat_dd],
        outputs=[var_dd, atk_dd] + _nav_outs,
    )
    cat_dd.change(
        on_category_change,
        inputs=[lang_radio, cat_dd],
        outputs=[var_dd, atk_dd] + _nav_outs,
    )
    var_dd.change(
        on_variant_change,
        inputs=[lang_radio, cat_dd, var_dd],
        outputs=[atk_dd] + _nav_outs,
    )
    atk_dd.change(
        on_attack_change,
        inputs=[lang_radio, cat_dd, var_dd, atk_dd],
        outputs=_nav_outs,
    )
    label_btn.click(
        on_save_label,
        inputs=[lang_radio, cat_dd, var_dd, atk_dd, label_radio],
        outputs=[verdict_html, judgment_md, label_radio, label_btn, label_status],
    )
    _hl_inputs = [lang_radio, cat_dd, var_dd, atk_dd, hl_input, color_radio, hl_state, raw_state]
    add_hl_btn.click(on_add_highlight,  inputs=_hl_inputs, outputs=_hl_outs)
    hl_input.submit( on_add_highlight,  inputs=_hl_inputs, outputs=_hl_outs)
    clear_hl_btn.click(
        on_clear_highlights,
        inputs=[lang_radio, cat_dd, var_dd, atk_dd, raw_state],
        outputs=[out_html_box, hl_state, hl_display],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=6002)
