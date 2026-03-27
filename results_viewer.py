"""
Gradio results viewer for VulnLLM-R evaluation results.
Groups samples by: language → category → variant → attack

Usage: python results_viewer.py
"""
import json
import re
from pathlib import Path
import gradio as gr

BASE = Path(__file__).parent

C_VARIANTS      = ["creatend", "findrec", "mkbuf", "allocate"]
PYTHON_VARIANTS = ["finduser", "makeconn", "parseitem", "loadconf"]
CATEGORIES      = ["clean", "dpi", "context_aware"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _build_fingerprint_map(lang: str, category: str, variant: str) -> dict:
    """Build code.strip()[-200:] → attack_name from dataset files."""
    lang_dir  = "Python" if lang == "Python" else "C"
    ds_subdir = "python" if lang == "Python" else "c"
    ds_dir = BASE / f"datasets/{lang_dir}/NPD/{category}/{variant}/{ds_subdir}"
    code_to_attack = {}
    if not ds_dir.exists():
        return code_to_attack
    for dsf in sorted(ds_dir.glob("*.json")):
        try:
            ds = json.loads(dsf.read_text())
            code = ds[0]["code"] if isinstance(ds, list) and ds else ""
            attack_name = dsf.stem  # e.g. findrec_COMP_1
            if attack_name.startswith(variant + "_"):
                attack_name = attack_name[len(variant) + 1:]
            code_to_attack[code.strip()[-200:]] = attack_name
        except Exception:
            pass
    return code_to_attack


def _load_dataset_code(lang: str, category: str, variant: str, attack: str) -> str:
    """Load the code field from the dataset JSON file."""
    lang_dir  = "Python" if lang == "Python" else "C"
    ds_subdir = "python" if lang == "Python" else "c"
    path = BASE / f"datasets/{lang_dir}/NPD/{category}/{variant}/{ds_subdir}/{variant}_{attack}.json"
    if not path.exists():
        return f"(dataset file not found: {path})"
    try:
        ds = json.loads(path.read_text())
        code = ds[0]["code"] if isinstance(ds, list) and ds else ""
        return code
    except Exception as e:
        return f"(error reading dataset: {e})"


def load_data(lang: str, category: str) -> dict:
    """
    Returns dict: variant → {attack → {flag, code, output,
                                        predicted_is_vulnerable,
                                        predicted_vulnerability_type,
                                        completion_tokens}}
    """
    lang_dir = "Python" if lang == "Python" else "C"
    result_dir = BASE / f"results/{lang_dir}/NPD/policy/{category}"
    variants = PYTHON_VARIANTS if lang == "Python" else C_VARIANTS

    out = {v: {} for v in variants}

    if not result_dir.exists():
        return out

    for result_file in sorted(result_dir.glob("*.json")):
        fname = result_file.name
        # Determine variant from filename
        if lang == "Python":
            m = re.search(rf'datasets_Python_NPD_{category}_([a-z]+)__', fname)
        else:
            m = re.search(rf'NPD_{category}_([a-z]+)__', fname)
        if not m:
            continue
        variant = m.group(1)
        if variant not in variants:
            continue

        # Build fingerprint map for this variant
        fp_map = _build_fingerprint_map(lang, category, variant)

        try:
            records = json.loads(result_file.read_text())
        except Exception:
            continue

        for rec in records:
            if "input" not in rec:
                continue
            inp = rec.get("input", "")
            # Extract code from first ``` block
            cs = inp.find("```")
            if cs == -1:
                code_snippet = ""
            else:
                cs += 3
                ce = inp.find("```", cs)
                code_snippet = inp[cs:ce].strip() if ce > cs else ""

            attack = fp_map.get(code_snippet.strip()[-200:], "?")
            out[variant][attack] = {
                "flag":                        rec.get("flag", "?"),
                "code":                        code_snippet,
                "output":                      rec.get("output", ""),
                "predicted_is_vulnerable":     rec.get("predicted_is_vulnerable", "?"),
                "predicted_vulnerability_type": rec.get("predicted_vulnerability_type", "?"),
                "completion_tokens":           rec.get("completion_tokens", "?"),
            }

    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def flag_badge_html(flag: str) -> str:
    styles = {
        "tp": ("✓ DETECTED",  "#d4edda", "#155724"),
        "fn": ("✗ EVADED",    "#f8d7da", "#721c24"),
        "fp": ("⚠ FALSE POS", "#fff3cd", "#856404"),
        "tn": ("✓ CLEAN",     "#d4edda", "#155724"),
    }
    label, bg, fg = styles.get(flag, (f"? {flag.upper()}", "#e2e3e5", "#383d41"))
    return (f'<span style="background:{bg};color:{fg};padding:6px 16px;'
            f'border-radius:4px;font-weight:bold;font-size:1.1em">{label}</span>')


def flag_badge(flag: str) -> str:
    return {
        "tp": "**TP** — detected (true positive)",
        "fn": "**FN** — missed (attack succeeded)",
        "fp": "**FP** — false alarm (false positive)",
        "tn": "**TN** — correctly clean (true negative)",
    }.get(flag, f"**{flag}**")


# ── Global data state ─────────────────────────────────────────────────────────

DATA         = {}
CURRENT_LANG = "C"
CURRENT_CAT  = "clean"


def _reload(lang: str, category: str):
    global DATA, CURRENT_LANG
    CURRENT_LANG = lang
    DATA = load_data(lang, category)


_reload("C", "clean")


# ── UI helpers ────────────────────────────────────────────────────────────────

def get_variants_for_data():
    return [v for v in DATA if DATA[v]]


def get_attacks_for_variant(variant: str):
    if variant not in DATA:
        return []
    return sorted(DATA[variant].keys())


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _display(lang: str, variant, attack):
    if not variant or not attack or variant not in DATA or attack not in DATA.get(variant, {}):
        return "(no data)", "", "", ""

    rec  = DATA[variant][attack]
    flag = rec.get("flag", "?")

    # Load code from dataset file (not extracted from input); fall back to stored snippet
    code = _load_dataset_code(lang, CURRENT_CAT, variant, attack)
    if code.startswith("(dataset file not found"):
        code = rec.get("code", "")

    model_out = rec.get("output", "")

    badge = flag_badge_html(flag)

    judgment = "\n".join([
        flag_badge(flag),
        f"",
        f"**Predicted vulnerable**: `{rec.get('predicted_is_vulnerable', '?')}`",
        f"**Predicted type**: `{rec.get('predicted_vulnerability_type', '?')}`",
        f"**Completion tokens**: {rec.get('completion_tokens', '?')}",
    ])

    return code, model_out, badge, judgment


def on_lang_or_cat_change_v2(lang: str, category: str):
    global CURRENT_CAT
    CURRENT_CAT = category
    _reload(lang, category)
    variants = get_variants_for_data()
    variant  = variants[0] if variants else None
    attacks  = get_attacks_for_variant(variant) if variant else []
    attack   = attacks[0] if attacks else None

    code, model_out, badge, judgment = _display(lang, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=variants, value=variant),
        gr.update(choices=attacks,  value=attack),
        gr.update(value=code, language=code_lang),
        model_out,
        badge,
        judgment,
    )


def on_variant_change_v2(lang: str, category: str, variant: str):
    global CURRENT_CAT
    CURRENT_CAT = category
    _reload(lang, category)
    attacks = get_attacks_for_variant(variant)
    attack  = attacks[0] if attacks else None
    code, model_out, badge, judgment = _display(lang, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return (
        gr.update(choices=attacks, value=attack),
        gr.update(value=code, language=code_lang),
        model_out,
        badge,
        judgment,
    )


def on_attack_change_v2(lang: str, category: str, variant: str, attack: str):
    global CURRENT_CAT
    CURRENT_CAT = category
    _reload(lang, category)
    code, model_out, badge, judgment = _display(lang, variant, attack)
    code_lang = "python" if lang == "Python" else "c"
    return gr.update(value=code, language=code_lang), model_out, badge, judgment


# ── Initial state ─────────────────────────────────────────────────────────────

init_lang     = "C"
init_cat      = "clean"
CURRENT_CAT   = init_cat
_reload(init_lang, init_cat)

init_variants = get_variants_for_data()
init_variant  = init_variants[0] if init_variants else None
init_attacks  = get_attacks_for_variant(init_variant) if init_variant else []
init_attack   = init_attacks[0] if init_attacks else None
init_code, init_out, init_badge, init_judgment = _display(init_lang, init_variant, init_attack)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="VulnLLM-R Results Viewer") as demo:
    gr.Markdown("# VulnLLM-R Results Viewer")
    gr.Markdown("Browse per-attack model outputs and detection outcomes.")

    with gr.Row():
        lang_radio = gr.Radio(choices=["C", "Python"], value=init_lang, label="Language")
        cat_dd     = gr.Dropdown(choices=CATEGORIES,    value=init_cat,      label="Category")
        var_dd     = gr.Dropdown(choices=init_variants, value=init_variant,  label="Variant")
        atk_dd     = gr.Dropdown(choices=init_attacks,  value=init_attack,   label="Attack")

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
    lang_radio.change(
        on_lang_or_cat_change_v2,
        inputs=[lang_radio, cat_dd],
        outputs=[var_dd, atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    cat_dd.change(
        on_lang_or_cat_change_v2,
        inputs=[lang_radio, cat_dd],
        outputs=[var_dd, atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    var_dd.change(
        on_variant_change_v2,
        inputs=[lang_radio, cat_dd, var_dd],
        outputs=[atk_dd, code_box, out_box, verdict_html, judgment_md],
    )
    atk_dd.change(
        on_attack_change_v2,
        inputs=[lang_radio, cat_dd, var_dd, atk_dd],
        outputs=[code_box, out_box, verdict_html, judgment_md],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=6002)
