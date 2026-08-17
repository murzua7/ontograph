"""Self-contained 3D force-graph HTML viewer for ontograph knowledge graphs.

Default ontograph visualization (as of 2026-04-21). Emits a single HTML file
with the KG payload inlined; library is pulled from CDN (jsdelivr → unpkg
fallback). The Cytoscape.js 2D viewer remains available via `--legacy-2d`.

Nodes are coloured by `entity_type`; edges are coloured by `sign`
(+ green, - red, ± orange, 0 grey, unknown faint). Animated particles on
signed edges reveal causal direction at a glance. The right-hand
encyclopedia panel doubles as a filterable search and detail pane: clicking
a node reveals its aliases, observations, code anchors, and in/out edges.
"""
from __future__ import annotations

import html as _html
import json
from collections import Counter
from pathlib import Path

from ontograph.clustering import cluster_kg as _cluster_kg
from ontograph.cycles import (
    MAX_CYCLES as _MAX_CYCLES,
    MAX_CYCLE_LEN as _MAX_CYCLE_LEN,
    enumerate_cycles as _enumerate_cycles,
)
from ontograph.models import KnowledgeGraph as _KnowledgeGraph
from ontograph.paths import sign_product

# Default threshold above which `cluster_mode='auto'` activates. Matches
# the heuristic from #47's spec: 3d-force-graph's stabiliser gets sluggish
# in the 3–5k-node range and visually unintelligible well before that,
# so coarsen before the user notices.
CLUSTER_THRESHOLD = 5000

SIGN_COLOR = {
    "+":       "#2ecc71",  # green
    "-":       "#e74c3c",  # red
    "±":       "#f39c12",  # orange
    "0":       "#7f8c8d",  # grey
    "unknown": "#34495e22",  # near-transparent dark
}
SIGN_PARTICLE = {"+": 2, "-": 2, "±": 2, "0": 0, "unknown": 0}

# Diff overlay palette (#44). Used only when a base KG is supplied to
# `build_3d_html`; otherwise every node/link is `unchanged` and the legend
# section is suppressed.
DIFF_COLOR = {
    "added":     "#3be083",  # green
    "removed":   "#ff6b6b",  # red
    "changed":   "#f5b44a",  # amber
    "unchanged": "#7a8a99",  # muted grey (neutral, not drawing the eye)
}

# Fields that define a relation's semantic payload for the change-detector.
# Mirrors `ontograph.diff._RELATION_FIELDS_WATCHED` — kept local so the
# viewer doesn't leak dataclass types into its raw-dict pipeline.
_REL_DIFF_FIELDS = ("sign", "lag", "edge_class", "form", "weight", "confidence")

# Categorical palette for entity types — Okabe-Ito + extensions for >8 types
PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#A65628", "#984EA3", "#FF7F00", "#FFFF33",
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666",
]


def _rel_key(r: dict) -> tuple[str, str, str]:
    return (r.get("source", ""), r.get("target", ""), r.get("relation_type", ""))


def _compute_diff(base_kg: dict, head_kg: dict) -> dict:
    """Classify entities and relations between two raw-dict KGs (#44).

    Returns a structure the viewer pipeline uses to stamp `diff_status` on
    every node/link and to surface the per-bucket count in the legend.
    Parallels `ontograph.diff.diff_kgs` but operates on raw dicts so the
    viewer payload never has to round-trip through dataclasses.
    """
    base_entities = base_kg.get("entities", {}) or {}
    head_entities = head_kg.get("entities", {}) or {}
    base_names, head_names = set(base_entities), set(head_entities)

    entity_status: dict[str, str] = {}
    for name in head_names - base_names:
        entity_status[name] = "added"
    for name in base_names - head_names:
        entity_status[name] = "removed"
    for name in head_names & base_names:
        b = base_entities[name] or {}
        h = head_entities[name] or {}
        if (b.get("entity_type") or "unknown") != (h.get("entity_type") or "unknown"):
            entity_status[name] = "changed"
        else:
            entity_status[name] = "unchanged"

    base_relations = base_kg.get("relations", []) or []
    head_relations = head_kg.get("relations", []) or []

    base_rel_by_key: dict[tuple[str, str, str], dict] = {}
    for r in base_relations:
        base_rel_by_key.setdefault(_rel_key(r), r)
    head_rel_by_key: dict[tuple[str, str, str], dict] = {}
    for r in head_relations:
        head_rel_by_key.setdefault(_rel_key(r), r)

    rel_status: dict[tuple[str, str, str], str] = {}
    for k in head_rel_by_key.keys() - base_rel_by_key.keys():
        rel_status[k] = "added"
    for k in base_rel_by_key.keys() - head_rel_by_key.keys():
        rel_status[k] = "removed"
    for k in head_rel_by_key.keys() & base_rel_by_key.keys():
        b = base_rel_by_key[k]
        h = head_rel_by_key[k]
        if any(b.get(f) != h.get(f) for f in _REL_DIFF_FIELDS):
            rel_status[k] = "changed"
        else:
            rel_status[k] = "unchanged"

    removed_entity_names = [n for n, s in entity_status.items() if s == "removed"]
    removed_rels = [base_rel_by_key[k] for k, s in rel_status.items() if s == "removed"]

    counts = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    for s in entity_status.values():
        counts[s] += 1
    rel_counts = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    for s in rel_status.values():
        rel_counts[s] += 1

    return {
        "entity_status": entity_status,
        "rel_status": rel_status,
        "removed_entities": {n: base_entities[n] for n in removed_entity_names},
        "removed_relations": removed_rels,
        "counts": {"entities": counts, "relations": rel_counts},
    }


def build_3d_html(
    kg_path: Path,
    out_path: Path,
    *,
    base_kg_path: Path | None = None,
    cluster_mode: bool | str = "auto",
    cluster_threshold: int = CLUSTER_THRESHOLD,
) -> Path:
    """Render a KG JSON as a self-contained 3D viewer HTML file.

    When `base_kg_path` is provided, the viewer is rendered in diff mode:
    every node/link carries a `diff_status` and a Diff legend section
    exposes per-bucket toggle checkboxes. Removed entities and relations
    are injected into the payload so they can be ghost-rendered — without
    this, the user couldn't see anything that was deleted.

    `cluster_mode` controls community-coarsened rendering (#47):
    - `'auto'` (default): coarsen when `len(entities) > cluster_threshold`.
    - `True`: always coarsen.
    - `False`: never coarsen.

    In clustered mode the viewer shows one supernode per community and
    one aggregated bridge per ordered cluster-pair. Diff overlay is
    suppressed in clustered mode — the two semantics don't compose
    cleanly (a "changed" cluster is a poorly-defined concept).

    Returns the path written.
    """
    kg_path = Path(kg_path)
    out_path = Path(out_path)
    kg = json.loads(kg_path.read_text(encoding="utf-8"))

    entities = dict(kg.get("entities", {}))
    relations = list(kg.get("relations", []))

    diff_info: dict | None = None
    if base_kg_path is not None:
        base_kg = json.loads(Path(base_kg_path).read_text(encoding="utf-8"))
        diff_info = _compute_diff(base_kg, kg)
        # Inject removed entities/relations into the rendered payload so the
        # viewer can show what was deleted. They live in the legend as
        # ghost-styled removed-bucket items.
        for n, e in diff_info["removed_entities"].items():
            entities.setdefault(n, e)
        relations = relations + list(diff_info["removed_relations"])

    # Clustering decision. `auto` activates above the threshold; explicit
    # bool overrides. Diff mode suppresses clustering because the two
    # overlays do not compose coherently.
    if diff_info is not None:
        cluster_active = False
    elif cluster_mode == "auto":
        cluster_active = len(entities) > cluster_threshold
    else:
        cluster_active = bool(cluster_mode)

    cluster_info: dict | None = None
    if cluster_active:
        kg_obj = _KnowledgeGraph.from_json(kg_path.read_text(encoding="utf-8"))
        cluster_info = _cluster_kg(kg_obj, max_nodes=0)

    types_in_use = sorted({e.get("entity_type", "unknown") for e in entities.values()})
    type_color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types_in_use)}

    degree: dict[str, int] = {}
    for r in relations:
        degree[r["source"]] = degree.get(r["source"], 0) + 1
        degree[r["target"]] = degree.get(r["target"], 0) + 1

    entity_status = (diff_info or {}).get("entity_status", {})
    rel_status = (diff_info or {}).get("rel_status", {})

    if cluster_info is not None:
        # Supernode payload. Each cluster becomes one node; each cluster
        # pair with cross-edges becomes one link. The individual entities
        # and relations are still carried in `src_entities` / `src_relations`
        # so the export code (#42) can recover the full KG from the viewer.
        cluster_summary = cluster_info["cluster_summary"]
        cluster_ecolor = "#9b8cff"  # muted violet — distinct from type palette
        nodes = []
        for cid, info in cluster_summary.items():
            members = info["members"]
            # Supernode size scales with sqrt(member count) for roughly
            # area-proportional rendering in force-graph's mass model.
            nodes.append({
                "id": cid,
                "type": "cluster",
                "color": cluster_ecolor,
                "type_color": cluster_ecolor,
                "val": 1 + (info["size"]) ** 0.5,
                "is_cluster": True,
                "size": info["size"],
                "members": members,
                "obs": f"{info['size']} entities",
                "obs_all": members[:8],
                "aliases": [],
                "anchors": [],
                "conf": None,
                "diff_status": "unchanged",
            })
        links = []
        for idx, e in enumerate(cluster_info["cluster_edges"]):
            links.append({
                "source": e["source"],
                "target": e["target"],
                "rel": "cluster_bridge",
                "sign": "unknown",
                "lag": "unknown",
                "form": "unknown",
                "ec": "cluster",
                "conf": None,
                "cond_on": [],
                "passage": "",
                "doc": "", "section": "", "page": None,
                "color": cluster_ecolor,
                "sign_color": SIGN_COLOR["unknown"],
                "pc": 0,
                "origIdx": idx,
                "diff_status": "unchanged",
                "count": e["count"],
                "is_cluster_edge": True,
            })
    else:
        nodes = []
        for name, e in entities.items():
            etype = e.get("entity_type", "unknown")
            obs_list = [o for o in (e.get("observations") or []) if o]
            aliases = [a for a in (e.get("aliases") or []) if a]
            anchors = e.get("code_anchors") or []
            status = entity_status.get(name, "unchanged")
            # In diff mode the diff-status signal is more informative than
            # entity-type clustering, so we swap the render colour. The
            # type-colour remains available via `n.type_color` so the legend
            # swatches and a future "show-as-type" toggle can still use it.
            render_color = DIFF_COLOR[status] if diff_info is not None else type_color[etype]
            nodes.append({
                "id": name,
                "type": etype,
                "color": render_color,
                "type_color": type_color[etype],
                "val": 1 + (degree.get(name, 0)) ** 0.5,
                "obs": (obs_list[0][:200] if obs_list else ""),
                "obs_all": obs_list[:8],
                "aliases": aliases,
                "anchors": [
                    f"{a.get('repo','')}/{a.get('path','')}:{a.get('line','')}"
                    for a in anchors[:4] if a.get('path')
                ],
                "conf": e.get("confidence"),
                "diff_status": status,
            })

        links = []
        for idx, r in enumerate(relations):
            sign = r.get("sign", "unknown") or "unknown"
            # Pull the first provenance entry for source-passage display.
            # Unlike DocAnchors (entity-level), Relation.provenance is already
            # indexed per-edge, so we just surface the first one.
            prov_list = r.get("provenance") or []
            prov0 = prov_list[0] if prov_list else {}
            status = rel_status.get(_rel_key(r), "unchanged")
            sign_color = SIGN_COLOR.get(sign, SIGN_COLOR["unknown"])
            render_color = DIFF_COLOR[status] if diff_info is not None else sign_color
            links.append({
                "source": r["source"],
                "target": r["target"],
                "rel": r.get("relation_type", ""),
                "sign": sign,
                "lag": r.get("lag", "unknown"),
                "form": r.get("form", "unknown"),
                "ec":   r.get("edge_class", "unknown"),
                "conf": r.get("confidence"),
                "cond_on": r.get("conditional_on") or [],
                "passage": (prov0.get("passage") or "")[:500],
                "doc":     prov0.get("document", ""),
                "section": prov0.get("section", ""),
                "page":    prov0.get("page"),
                "color": render_color,
                "sign_color": sign_color,
                "pc":   SIGN_PARTICLE.get(sign, 0),
                # Index into the original `relations` list — used by the
                # subgraph-export feature (#42) to rebuild a lossless KG from
                # whatever edges the viewer is currently showing.
                "origIdx": idx,
                "diff_status": status,
            })

    cycles = _enumerate_cycles(relations)

    # `src_entities` and `src_relations` carry the untransformed KG so that
    # the JS export code (#42) can re-emit a canonical ontograph JSON for
    # the currently-visible subgraph — lossless wrt. the viewer inputs.
    data_json = json.dumps(
        {
            "nodes": nodes, "links": links, "cycles": cycles,
            "src_entities": entities, "src_relations": relations,
        },
        ensure_ascii=False,
    )

    # Legend rows double as filter toggles. Each row is a checkbox whose
    # real value (possibly containing non-id-safe chars) is carried on a
    # `data-val` attribute and consulted by `applyFilters()`.
    n_edges = len(links)
    n_nodes = len(nodes)
    sign_counts = Counter(l["sign"] for l in links)
    ec_counts = Counter(l["ec"] or "unknown" for l in links)
    type_counts = Counter(n["type"] for n in nodes)
    signed = sum(v for k, v in sign_counts.items() if k != "unknown")
    title = kg_path.stem

    def _esc(s: str) -> str:
        return _html.escape(str(s), quote=True)

    type_legend = "".join(
        f'<div class="row"><label class="tgRow">'
        f'<input type="checkbox" class="tgType" data-val="{_esc(t)}" checked>'
        f'<span class="sw" style="background:{c}"></span>'
        f'<span class="lbl">{_esc(t)}</span>'
        f'<span class="ct">{type_counts.get(t, 0)}</span>'
        f'</label></div>'
        for t, c in type_color.items()
    )
    sign_order = [("+", "+ (pos)"), ("-", "- (neg)"), ("±", "± (mixed)"),
                  ("0", "0 (zero)"), ("unknown", "unknown")]
    sign_legend = "".join(
        f'<div class="row"><label class="tgRow">'
        f'<input type="checkbox" class="tgSign" data-val="{_esc(s)}" checked>'
        f'<span class="sw" style="background:{SIGN_COLOR[s]}"></span>'
        f'<span class="lbl">{_esc(label)}</span>'
        f'<span class="ct">{sign_counts.get(s, 0)}</span>'
        f'</label></div>'
        for s, label in sign_order if sign_counts.get(s)
    )
    edge_class_legend = "".join(
        f'<div class="row"><label class="tgRow">'
        f'<input type="checkbox" class="tgEc" data-val="{_esc(ec)}" checked>'
        f'<span class="lbl">{_esc(ec)}</span>'
        f'<span class="ct">{ec_counts.get(ec, 0)}</span>'
        f'</label></div>'
        for ec in sorted(ec_counts)
    )

    # Diff legend — only rendered when a base KG was supplied. Each row is
    # "N ent / M rel" so the user sees both slices of the diff at a glance.
    if diff_info is not None:
        ec = diff_info["counts"]["entities"]
        rc = diff_info["counts"]["relations"]
        diff_rows = []
        for status in ("added", "removed", "changed", "unchanged"):
            diff_rows.append(
                f'<div class="row"><label class="tgRow">'
                f'<input type="checkbox" class="tgDiff" data-val="{status}" checked>'
                f'<span class="sw" style="background:{DIFF_COLOR[status]}"></span>'
                f'<span class="lbl">{status}</span>'
                f'<span class="ct">{ec[status]}n / {rc[status]}e</span>'
                f'</label></div>'
            )
        diff_legend = (
            '<div class="sep"></div>'
            '<h3>Diff <span class="hint">(toggle to filter)</span></h3>'
            f'<div id="diffSummary" style="font-size:10px;color:#8faccb;margin-bottom:4px">'
            f'+{ec["added"]} / -{ec["removed"]} / ~{ec["changed"]} entities · '
            f'+{rc["added"]} / -{rc["removed"]} / ~{rc["changed"]} relations</div>'
            + "".join(diff_rows)
        )
    else:
        diff_legend = ""

    if cluster_info is not None:
        orig_n = len(kg.get("entities", {}))
        orig_e = len(kg.get("relations", []))
        cluster_banner = (
            f'<div class="stat" id="cluster-mode" '
            f'style="color:#9b8cff;font-size:10px;margin-top:4px">'
            f'Clustered: {len(nodes)} supernodes '
            f'({orig_n} entities, {orig_e} relations)</div>'
        )
        header_counts_line = (
            f'<span class="stat">{n_nodes}</span> supernodes · '
            f'<span class="stat">{n_edges}</span> bridges'
        )
    else:
        cluster_banner = ""
        header_counts_line = (
            f'<span class="stat">{n_nodes}</span> entities · '
            f'<span class="stat">{n_edges}</span> relations'
        )

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title} — 3D KG</title>
<style>
  html,body {{ margin:0; padding:0; background:#0b0d10; color:#ddd;
               font-family: -apple-system, system-ui, sans-serif; overflow:hidden; }}
  #graph {{ width:100vw; height:100vh; }}
  .panel {{
    position:absolute; background:rgba(12,14,18,0.85); padding:10px 12px;
    border:1px solid #2a3038; border-radius:6px; font-size:11px; line-height:1.4;
    max-height: 70vh; overflow-y:auto; backdrop-filter: blur(4px);
  }}
  #legend  {{ top:12px; left:12px; max-width: 260px; }}
  #encyc   {{ top:12px; right:12px; width:288px; bottom:188px; padding:10px 12px;
             display:flex; flex-direction:column; gap:6px; }}
  #header  {{ right:12px; bottom:12px; width:288px; text-align:left; font-size:10.5px; }}
  #hover   {{ bottom:12px; left:12px; right:324px; max-width:none; max-height:160px;
             font-family: ui-monospace, Menlo, monospace; font-size:11px; }}
  #edgeInfo {{
    bottom: 180px; left:12px; right:324px; max-width:none; max-height:200px;
    font-family: ui-monospace, Menlo, monospace; font-size:11px; display:none;
    border-left: 3px solid #7fc5ff;
  }}
  #edgeInfo .edgeHead {{
    display:flex; justify-content:space-between; align-items:baseline;
    margin-bottom:4px; font-size:11px;
  }}
  #edgeInfo .edgeHead .rel {{
    color:#fff; font-weight:600;
  }}
  #edgeInfo .edgeHead .facets {{ color:#7fc5ff; font-size:10px; }}
  #edgeInfo .edgeHead .close {{
    background:#2a3038; border:none; color:#ddd;
    padding:2px 8px; border-radius:3px; cursor:pointer; font-size:10px;
  }}
  #edgeInfo .passage {{
    padding:6px 8px; background:#0e141b; border-radius:3px;
    color:#cfdceb; font-size:10.5px; line-height:1.45;
    border-left:2px solid #2a4054;
    max-height:100px; overflow-y:auto;
    white-space:pre-wrap;
  }}
  #edgeInfo .cite {{
    margin-top: 4px; font-size: 9.5px; color:#6a90b5;
  }}
  #edgeInfo .cite code {{
    color:#8faccb; font-family: ui-monospace, Menlo, monospace;
  }}
  #edgeInfo .nopass {{
    color:#6a7a8a; font-style:italic; padding:4px 0;
  }}
  #pathInfo {{
    bottom: 400px; left:12px; right:324px; max-width:none; max-height:180px;
    font-family: ui-monospace, Menlo, monospace; font-size:11px; display:none;
    border-left: 3px solid #3be083;
  }}
  #pathInfo .phead {{
    display:flex; justify-content:space-between; align-items:baseline;
    margin-bottom:4px;
  }}
  #pathInfo .phead .close {{
    background:#2a3038; border:none; color:#ddd;
    padding:2px 8px; border-radius:3px; cursor:pointer; font-size:10px;
  }}
  #pathInfo .pchain {{
    padding:6px 8px; background:#0e141b; border-radius:3px;
    color:#cfdceb; font-size:10.5px; line-height:1.75;
    border-left:2px solid #26402f;
    max-height:90px; overflow-y:auto;
    word-break: break-all;
  }}
  #pathInfo .pchain .pnode {{
    color:#fff; padding:0 3px; background:#1a2a3a; border-radius:2px;
    font-weight:600;
  }}
  #pathInfo .pchain .pedge {{ font-weight:600; }}
  #pathInfo .pprod {{ margin-top:4px; font-size:10px; color:#aaa; }}
  #pathInfo .nopath {{ color:#cc8080; font-style:italic; padding:4px 0; }}
  #pathCtl {{ margin-top:4px; }}
  #pathCtl .row {{ justify-content: space-between; gap: 4px; margin: 3px 0; }}
  #pathCtl button {{
    background:#1a2a3a; border:1px solid #2a4054; color:#bcd6ee;
    padding:2px 6px; border-radius:3px; font-size:10px; cursor:pointer;
    font-family: inherit;
  }}
  #pathCtl button:hover {{ background:#24364a; }}
  #pathCtl button:disabled {{ opacity:0.4; cursor:default; }}
  #pathCtl code {{
    color:#fff; font-family: ui-monospace, Menlo, monospace; font-size:10px;
  }}
  #pathCtl .lbl {{ color:#7fc5ff; font-size:10px; }}
  #cycleBox {{ margin-top:4px; max-height: 180px; overflow-y:auto; }}
  #cycleBox::-webkit-scrollbar {{ width:6px; }}
  #cycleBox::-webkit-scrollbar-track {{ background:#0e141b; border-radius:3px; }}
  #cycleBox::-webkit-scrollbar-thumb {{ background:#24435e; border-radius:3px; }}
  #cycleBox .cyc {{
    display:flex; align-items:center; gap:6px;
    padding:3px 4px; cursor:pointer; border-radius:3px;
    font-family: ui-monospace, Menlo, monospace; font-size:10px;
    line-height:1.3;
  }}
  #cycleBox .cyc:hover {{ background:#182028; }}
  #cycleBox .cyc.active {{ background:#2a1e2e; outline:1px solid #c9a0ff; }}
  #cycleBox .cyc .lag {{
    flex:0 0 auto; padding:0 4px; border-radius:2px;
    font-weight:600; font-size:9px;
  }}
  #cycleBox .cyc .lag.w {{ background:#1a2a3a; color:#7fc5ff; }}
  #cycleBox .cyc .lag.a {{ background:#2a1e2e; color:#c9a0ff; }}
  #cycleBox .cyc .prod {{
    flex:0 0 auto; font-weight:600; padding:0 3px;
  }}
  #cycleBox .cyc .chain {{
    flex:1; color:#bcd6ee; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap;
  }}
  #cycleBox .empty {{
    color:#6a7a8a; font-style:italic; padding:4px 0; font-size:10px;
  }}
  #searchInput {{
    width:100%; box-sizing:border-box;
    background: linear-gradient(180deg, #0b1824 0%, #0d1a28 100%);
    color:#e8f4ff; border:1px solid #24435e;
    border-bottom: 2px solid #7fc5ff;
    border-radius:5px; padding:8px 10px 8px 30px; font-size:12px;
    outline:none; font-family: inherit;
    letter-spacing:0.15px;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.4), 0 0 0 1px rgba(127,197,255,0.04);
    transition: box-shadow 180ms, border-color 180ms;
  }}
  #searchInput::placeholder {{ color:#6b8aa8; font-style:italic; }}
  #searchInput:focus {{
    border-color:#7fc5ff; border-bottom-color:#9ee4ff;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.4),
                0 0 0 1px rgba(127,197,255,0.35),
                0 0 18px rgba(127,197,255,0.25);
  }}
  #searchBox {{ position:relative; flex: 0 0 auto; }}
  #searchBox::before {{
    content:""; position:absolute; left:9px; top:50%;
    width:14px; height:14px; transform: translateY(-55%);
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%237fc5ff' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'><circle cx='10.5' cy='10.5' r='7'/><line x1='21' y1='21' x2='15.8' y2='15.8'/></svg>");
    background-size: contain; background-repeat: no-repeat;
    opacity: 0.95; pointer-events:none;
  }}
  #selectedInfo {{
    display:none; padding:6px 4px 6px 4px; border-top:1px solid #2a3038;
    border-bottom:1px solid #2a3038; font-size:11px;
    max-height: 30vh; overflow-y:auto; flex: 0 0 auto;
  }}
  #selectedInfo h4 {{ margin:0 0 4px 0; font-size:11px; color:#7fc5ff; }}
  #selectedInfo .nb {{ color:#ddd; padding:1px 0; font-family:ui-monospace,Menlo,monospace; }}
  #selectedInfo .nb .arr {{ color:#888; margin:0 4px; }}
  #selDesc {{ margin:4px 0 8px 0; }}
  #selDesc .aka {{
    margin: 4px 0 6px 0; color:#8aa5bf; font-size:10px;
  }}
  #selDesc .aka .tag {{
    display:inline-block; margin-right:4px; padding:1px 5px;
    background:#1a2a3a; border:1px solid #2a4054; border-radius:3px;
    color:#bcd6ee; font-family:ui-monospace,Menlo,monospace; font-size:10px;
  }}
  #selDesc ul.obs {{
    margin: 4px 0 6px 0; padding-left: 14px; color:#cfdceb; font-size:10.5px;
    line-height:1.45;
  }}
  #selDesc ul.obs li {{ margin: 2px 0; }}
  #selDesc .anchors {{
    margin-top: 6px; padding-top:5px; border-top:1px dashed #243345;
    font-size: 9.5px; color:#6a90b5;
  }}
  #selDesc .anchors code {{
    font-family:ui-monospace,Menlo,monospace; color:#8faccb;
  }}
  #selDesc .noobs {{ color:#6a7a8a; font-style:italic; font-size:10px; padding:2px 0; }}
  #selectedInfo .clear {{
    float:right; background:#2a3038; border:none; color:#ddd;
    padding:2px 8px; border-radius:3px; cursor:pointer; font-size:10px;
  }}
  /* Encyclopedia */
  #encyc h3 {{ display:flex; justify-content:space-between; align-items:baseline; }}
  #encycList {{
    margin-top:6px; overflow-y:auto; height:calc(100% - 34px);
    font-size:11px;
  }}
  #encycList .letter {{
    position:sticky; top:0; background:rgba(12,14,18,0.95);
    color:#7fc5ff; font-weight:700; font-size:11px;
    padding:6px 2px 3px 2px; border-bottom:1px solid #223043;
    margin-top:6px; letter-spacing:1px;
  }}
  #encycList .letter:first-child {{ margin-top:0; }}
  #encycList .e {{
    display:flex; align-items:center; gap:6px;
    padding:3px 4px; cursor:pointer; border-radius:3px;
    line-height:1.3;
  }}
  #encycList .e:hover {{ background:#182028; }}
  #encycList .e.active {{ background:#1e3244; outline:1px solid #7fc5ff; }}
  #encycList .e .dot {{
    flex:0 0 8px; width:8px; height:8px; border-radius:50%;
    box-shadow: 0 0 4px rgba(0,0,0,0.5);
  }}
  #encycList .e .n {{
    flex:1; font-family: ui-monospace, Menlo, monospace;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#ddd;
  }}
  #encycList .e .t {{
    flex:0 0 auto; font-size:9.5px; color:#6a90b5;
    max-width:80px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }}
  /* Scrollbar polish — applied uniformly to every scrollable panel/box */
  .panel,
  #edgeInfo, #pathInfo, #selectedInfo,
  #cycleBox, #encycList {{
    scrollbar-width: thin;
    scrollbar-color: #24435e #0e141b;
  }}
  .panel::-webkit-scrollbar,
  #edgeInfo::-webkit-scrollbar, #pathInfo::-webkit-scrollbar,
  #selectedInfo::-webkit-scrollbar,
  #encycList::-webkit-scrollbar {{ width:8px; }}
  .panel::-webkit-scrollbar-track,
  #edgeInfo::-webkit-scrollbar-track, #pathInfo::-webkit-scrollbar-track,
  #selectedInfo::-webkit-scrollbar-track,
  #encycList::-webkit-scrollbar-track {{ background: #0e141b; border-radius:4px; }}
  .panel::-webkit-scrollbar-thumb,
  #edgeInfo::-webkit-scrollbar-thumb, #pathInfo::-webkit-scrollbar-thumb,
  #selectedInfo::-webkit-scrollbar-thumb,
  #encycList::-webkit-scrollbar-thumb {{ background: #24435e; border-radius:4px; }}
  .panel::-webkit-scrollbar-thumb:hover,
  #edgeInfo::-webkit-scrollbar-thumb:hover, #pathInfo::-webkit-scrollbar-thumb:hover,
  #selectedInfo::-webkit-scrollbar-thumb:hover,
  #encycList::-webkit-scrollbar-thumb:hover {{ background: #3b6080; }}
  h3 {{ margin:0 0 6px 0; font-size:12px; font-weight:600; color:#fff; }}
  .row {{ display:flex; align-items:center; margin:2px 0; }}
  .sw  {{ display:inline-block; width:10px; height:10px; border-radius:2px;
         margin-right:6px; border:1px solid #222; }}
  .stat {{ color:#7fc5ff; font-weight:600; }}
  /* Checkbox-legend rows: swatch → label → count, right-aligned. */
  .tgRow {{ display:flex; align-items:center; gap:6px; cursor:pointer;
           flex:1; user-select:none; }}
  .tgRow input[type=checkbox] {{ margin:0; }}
  .tgRow .lbl {{ flex:1; }}
  .tgRow .ct {{ color:#6a8aaa; font-size:10px; font-variant-numeric:tabular-nums; }}
  .tgRow input[type=checkbox]:not(:checked) ~ .lbl {{ color:#6a7a8a; }}
  .tgRow input[type=checkbox]:not(:checked) ~ .sw {{ opacity:0.3; }}
  #legend h3 .hint {{ color:#6a8aaa; font-weight:400; font-size:10px; margin-left:4px; }}
  #exportCtl {{ gap:6px; }}
  #exportCtl button {{
    background:#1a2a3a; border:1px solid #2a4054; color:#bcd6ee;
    padding:3px 10px; border-radius:3px; cursor:pointer; font-size:10.5px;
  }}
  #exportCtl button:hover {{ background:#243648; }}
  #exportCtl #exportCt {{ margin-left:auto; }}
  .sep {{ height:1px; background:#2a3038; margin:8px 0; }}
  kbd {{ background:#1e2328; padding:1px 5px; border-radius:3px;
         border:1px solid #2a3038; font-size:10px; }}
</style>
</head><body>
<div id="graph"></div>

<div id="encyc" class="panel">
  <h3>Encyclopedia <span id="encycCount" style="color:#7fc5ff;font-weight:600"></span></h3>
  <div id="searchBox">
    <input id="searchInput" placeholder="Search entities  (↑↓ pick · Enter focus · Esc clear)" autocomplete="off" spellcheck="false">
  </div>
  <div id="selectedInfo">
    <button class="clear" id="clearSel">clear</button>
    <h4 id="selName">—</h4>
    <div id="selDesc"></div>
    <div id="selNeighbors"></div>
  </div>
  <div id="encycList"></div>
</div>

<div id="header" class="panel">
  <h3>{title}</h3>
  <div>{header_counts_line}</div>
  <div><span class="stat">{signed}</span> signed
       ({sign_counts.get('+',0)}+ · {sign_counts.get('-',0)}- · {sign_counts.get('±',0)}± · {sign_counts.get('0',0)}·0)</div>
  {cluster_banner}
  <div class="sep"></div>
  <div><kbd>drag</kbd> rotate · <kbd>scroll</kbd> zoom · <kbd>r-drag</kbd> pan</div>
  <div><kbd>click</kbd> focus · <kbd>hover</kbd> inspect</div>
</div>

<div id="legend" class="panel">
  <h3>Entity types <span class="hint">(toggle to filter)</span></h3>
  {type_legend}
  <div class="sep"></div>
  <h3>Edge sign <span class="hint">(toggle to filter)</span></h3>
  {sign_legend}
  <div class="sep"></div>
  <h3>Edge class <span class="hint">(toggle to filter)</span></h3>
  {edge_class_legend}
  {diff_legend}
  <div class="sep"></div>
  <h3>Filter</h3>
  <div class="row"><label><input type="checkbox" id="hideIsolated"> hide orphans</label></div>
  <div class="sep"></div>
  <h3>Export <span class="hint">(currently visible)</span></h3>
  <div class="row" id="exportCtl">
    <button id="btnExportJSON" title="Download visible subgraph as ontograph KG JSON (re-renderable)">JSON</button>
    <button id="btnExportCSV" title="Download visible edges as CSV (source,target,sign,lag,edge_class,relation_type)">CSV</button>
    <span id="exportCt" class="ct"></span>
  </div>
  <div class="sep"></div>
  <h3>Signed path</h3>
  <div id="pathCtl">
    <div class="row"><span class="lbl">source:</span>
      <code id="pathSrcLbl">—</code>
      <button id="btnUseSrc" title="Use currently selected node as source" disabled>use sel</button></div>
    <div class="row"><span class="lbl">target:</span>
      <code id="pathTgtLbl">—</code>
      <button id="btnUseTgt" title="Use currently selected node as target" disabled>use sel</button></div>
    <div class="row" style="justify-content:flex-end">
      <button id="btnPathClear">clear path</button>
    </div>
  </div>
  <div class="sep"></div>
  <h3>Cycles <span id="cycleCount" style="color:#7fc5ff;font-weight:600"></span></h3>
  <div id="cycleBox"></div>
</div>

<div id="hover" class="panel" style="display:none"></div>

<div id="edgeInfo" class="panel"></div>

<div id="pathInfo" class="panel"></div>

<div id="err" style="position:absolute;top:0;left:0;right:0;padding:8px;background:#902020;color:#fff;font-family:monospace;font-size:12px;display:none;white-space:pre-wrap;z-index:999;"></div>

<script>
  window.__errs = [];
  window.addEventListener('error', e => {{
    window.__errs.push(`${{e.message}}  @  ${{e.filename||'?'}}:${{e.lineno||'?'}}`);
    const el = document.getElementById('err');
    el.style.display = 'block';
    el.textContent = 'JS ERRORS:\\n' + window.__errs.join('\\n');
  }});
</script>

<!-- 3d-force-graph bundles three internally; loading three separately causes
     version conflicts. Fallback chain: jsdelivr → unpkg. -->
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.73.4/dist/3d-force-graph.min.js"
        onerror="this.onerror=null;var s=document.createElement('script');s.src='https://unpkg.com/3d-force-graph/dist/3d-force-graph.min.js';document.head.appendChild(s);"></script>

<script>
const DATA = {data_json};

function go() {{
  if (typeof ForceGraph3D === 'undefined') {{
    document.getElementById('err').style.display='block';
    document.getElementById('err').textContent =
      'ForceGraph3D library did not load. CDN may be blocked by your network.';
    return;
  }}

const elHover = document.getElementById('hover');

const nodeById = {{}};
DATA.nodes.forEach(n => {{ nodeById[n.id] = n; }});
const neighbors = {{}};
const adjLinks  = {{}};
DATA.links.forEach(l => {{
  const s = typeof l.source === 'string' ? l.source : l.source.id;
  const t = typeof l.target === 'string' ? l.target : l.target.id;
  (neighbors[s] ||= new Set()).add(t);
  (neighbors[t] ||= new Set()).add(s);
  (adjLinks[s]  ||= new Set()).add(l);
  (adjLinks[t]  ||= new Set()).add(l);
}});

let selectedId = null;

// ---- Signed shortest-path state ---------------------------------------
let pathState = {{ source: null, target: null, links: [] }};
const pathLinks = new Set(); // link object identities in the current path
const pathNodeIds = new Set();

// ---- Cycle explorer state ---------------------------------------------
let activeCycleIdx = null;           // index into DATA.cycles
const cycleLinks = new Set();        // link object identities for the selected cycle
const cycleNodeIds = new Set();

// Directed adjacency for BFS. Built once; `DATA.links` elements are shared
// between pre- and post-init (3d-force-graph replaces source/target with
// node objects after init, but the link objects themselves persist).
const dirAdj = {{}};
DATA.links.forEach(l => {{
  const s = (l.source && l.source.id) || l.source;
  const t = (l.target && l.target.id) || l.target;
  (dirAdj[s] ||= []).push({{ link: l, to: t }});
}});

function isHighlighted(nodeId) {{
  if (cycleLinks.size) return cycleNodeIds.has(nodeId);
  if (pathLinks.size) return pathNodeIds.has(nodeId);
  if (!selectedId) return true;
  if (nodeId === selectedId) return true;
  return neighbors[selectedId]?.has(nodeId);
}}
function isLinkHighlighted(l) {{
  if (cycleLinks.size) return cycleLinks.has(l);
  if (pathLinks.size) return pathLinks.has(l);
  if (!selectedId) return true;
  return adjLinks[selectedId]?.has(l);
}}

const Graph = ForceGraph3D()(document.getElementById('graph'))
  .graphData(DATA)
  .backgroundColor('#0b0d10')
  .nodeLabel(n => `<b>${{n.id}}</b><br><span style='color:#7fc5ff'>${{n.type}}</span><br>${{n.obs}}`)
  .nodeColor(n => isHighlighted(n.id) ? n.color : 'rgba(60,60,60,0.15)')
  .nodeVal(n => (n.id === selectedId ? n.val * 4 : n.val))
  .nodeOpacity(0.95)
  .linkColor(l => isLinkHighlighted(l) ? l.color : 'rgba(40,40,50,0.05)')
  .linkOpacity(0.75)
  .linkWidth(l => {{
    if (cycleLinks.has(l)) return 3.5;
    if (pathLinks.has(l)) return 3.5;
    if (selectedId && adjLinks[selectedId]?.has(l)) return 2.0;
    return l.sign === 'unknown' ? 0.2 : 0.7;
  }})
  .linkDirectionalArrowLength(l => isLinkHighlighted(l) ? 3.5 : 0)
  .linkDirectionalArrowRelPos(0.92)
  .linkDirectionalArrowColor(l => l.color)
  .linkDirectionalParticles(l => {{
    if (cycleLinks.has(l)) return 6;
    if (pathLinks.has(l)) return 6;
    if (selectedId) return adjLinks[selectedId]?.has(l) ? 4 : 0;
    return l.pc;
  }})
  .linkDirectionalParticleSpeed(0.004)
  .linkDirectionalParticleWidth(1.6)
  .linkLabel(l => `<div style='font-family:monospace'>${{l.source.id||l.source}}  <b>[${{l.rel}}]</b>  ${{l.target.id||l.target}}<br>sign=${{l.sign}}  lag=${{l.lag}}  form=${{l.form}}  class=${{l.ec}}</div>`)
  .onNodeHover(n => {{
    if (!n) {{ elHover.style.display='none'; return; }}
    elHover.style.display='block';
    elHover.innerHTML =
      `<b style='color:#fff'>${{n.id}}</b>  <span style='color:#7fc5ff'>${{n.type}}</span><br>` +
      `<span style='color:#aaa'>${{n.obs || ''}}</span>`;
  }})
  .onNodeClick(n => selectNode(n.id))
  .onLinkClick(l => selectEdge(l));

function focusCamera(n) {{
  const dist = 120;
  const dr = Math.hypot(n.x||0, n.y||0, n.z||0) || 1;
  Graph.cameraPosition(
    {{ x: n.x * (1 + dist/dr), y: n.y * (1 + dist/dr), z: n.z * (1 + dist/dr) }},
    n, 900
  );
}}

function refreshHighlight() {{
  Graph.nodeColor(Graph.nodeColor())
       .linkColor(Graph.linkColor())
       .linkWidth(Graph.linkWidth())
       .linkDirectionalArrowLength(Graph.linkDirectionalArrowLength())
       .linkDirectionalParticles(Graph.linkDirectionalParticles())
       .nodeVal(Graph.nodeVal());
}}

function selectNode(id) {{
  const n = nodeById[id];
  if (!n) return;
  selectedId = id;
  focusCamera(n);
  refreshHighlight();
  renderSelectedPanel();
}}

function clearSelection() {{
  selectedId = null;
  refreshHighlight();
  document.getElementById('selectedInfo').style.display = 'none';
}}

function renderSelectedPanel() {{
  const info = document.getElementById('selectedInfo');
  if (!selectedId) {{ info.style.display='none'; return; }}
  info.style.display = 'block';
  const n = nodeById[selectedId];
  document.getElementById('selName').innerHTML =
    `<span style='color:#fff'>${{escapeHtml(n.id)}}</span>  ` +
    `<span style='color:#7fc5ff;font-weight:normal'>${{n.type}}</span>`;
  const descDiv = document.getElementById('selDesc');
  const descRows = [];
  if (n.aliases && n.aliases.length) {{
    descRows.push(
      `<div class='aka'>aka ` +
      n.aliases.map(a => `<span class='tag'>${{escapeHtml(a)}}</span>`).join(' ') +
      `</div>`
    );
  }}
  if (n.obs_all && n.obs_all.length) {{
    descRows.push(`<ul class='obs'>` +
      n.obs_all.map(o => `<li>${{escapeHtml(o)}}</li>`).join('') +
      `</ul>`);
  }} else {{
    descRows.push(`<div class='noobs'>no description captured</div>`);
  }}
  if (n.anchors && n.anchors.length) {{
    descRows.push(
      `<div class='anchors'>` +
      n.anchors.map(a => `<code>${{escapeHtml(a)}}</code>`).join('<br>') +
      `</div>`
    );
  }}
  descDiv.innerHTML = descRows.join('');
  const nbDiv = document.getElementById('selNeighbors');
  const adj = adjLinks[selectedId] ? [...adjLinks[selectedId]] : [];
  const outgoing = adj.filter(l => (l.source.id || l.source) === selectedId);
  const incoming = adj.filter(l => (l.target.id || l.target) === selectedId);
  const rows = [];
  rows.push(`<div style='color:#888;margin:6px 0 2px 0;'>outgoing (${{outgoing.length}})</div>`);
  outgoing.slice(0, 40).forEach(l => {{
    const other = (l.target.id || l.target);
    rows.push(`<div class='nb'><span style='color:${{l.color}}'>${{l.sign}}</span> <span class='arr'>→</span> <a href='#' data-id='${{encodeURIComponent(other)}}'>${{escapeHtml(other)}}</a> <span style='color:#666'>[${{l.rel}}]</span></div>`);
  }});
  rows.push(`<div style='color:#888;margin:6px 0 2px 0;'>incoming (${{incoming.length}})</div>`);
  incoming.slice(0, 40).forEach(l => {{
    const other = (l.source.id || l.source);
    rows.push(`<div class='nb'><a href='#' data-id='${{encodeURIComponent(other)}}'>${{escapeHtml(other)}}</a> <span class='arr'>→</span> <span style='color:${{l.color}}'>${{l.sign}}</span> <span style='color:#666'>[${{l.rel}}]</span></div>`);
  }});
  nbDiv.innerHTML = rows.join('');
  nbDiv.querySelectorAll('a[data-id]').forEach(a => {{
    a.addEventListener('click', ev => {{
      ev.preventDefault();
      selectNode(decodeURIComponent(a.getAttribute('data-id')));
    }});
  }});
}}

function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}

// ---- Edge panel: click an edge to see its source passage + facets ------
const edgeInfoEl = document.getElementById('edgeInfo');

function selectEdge(l) {{
  if (!l) {{ edgeInfoEl.style.display='none'; return; }}
  const s = (l.source.id || l.source);
  const t = (l.target.id || l.target);
  const facets = [];
  facets.push(`sign=<span style="color:${{l.color}}">${{escapeHtml(l.sign)}}</span>`);
  facets.push(`lag=${{escapeHtml(l.lag || 'unknown')}}`);
  facets.push(`form=${{escapeHtml(l.form || 'unknown')}}`);
  facets.push(`class=${{escapeHtml(l.ec || 'unknown')}}`);
  if (typeof l.conf === 'number') facets.push(`conf=${{l.conf.toFixed(2)}}`);
  if (l.cond_on && l.cond_on.length) {{
    facets.push(`cond: ${{l.cond_on.slice(0,3).map(escapeHtml).join(', ')}}`);
  }}

  const hasPassage = l.passage && l.passage.trim().length > 0;
  const citeBits = [];
  if (l.doc) citeBits.push(`<code>${{escapeHtml(l.doc)}}</code>`);
  if (l.section) citeBits.push(`§ ${{escapeHtml(l.section)}}`);
  if (l.page != null && l.page !== '') citeBits.push(`p.${{escapeHtml(String(l.page))}}`);

  const bodyHtml = hasPassage
    ? `<div class="passage">${{escapeHtml(l.passage)}}</div>` +
      (citeBits.length ? `<div class="cite">${{citeBits.join(' · ')}}</div>` : '')
    : `<div class="nopass">No source passage recorded for this edge.</div>` +
      (citeBits.length ? `<div class="cite">${{citeBits.join(' · ')}}</div>` : '');

  edgeInfoEl.innerHTML =
    `<div class="edgeHead">` +
      `<span><span class="rel">${{escapeHtml(s)}}</span> ` +
      `<span style="color:#888">—[</span>` +
      `<span style="color:#7fc5ff">${{escapeHtml(l.rel || 'relation')}}</span>` +
      `<span style="color:#888">]→</span> ` +
      `<span class="rel">${{escapeHtml(t)}}</span></span>` +
      `<button class="close" onclick="document.getElementById('edgeInfo').style.display='none'">close</button>` +
    `</div>` +
    `<div class="facets" style="color:#7fc5ff;font-size:10px;margin-bottom:6px">${{facets.join(' · ')}}</div>` +
    bodyHtml;
  edgeInfoEl.style.display = 'block';
}}

// ---- Signed shortest path --------------------------------------------
// BFS on the directed link graph. Returns list of link objects or null.
function bfsPath(srcId, tgtId) {{
  if (srcId === tgtId) return [];
  const prev = Object.create(null);
  prev[srcId] = {{ link: null, from: null }};
  const q = [srcId];
  let head = 0;
  while (head < q.length) {{
    const u = q[head++];
    if (u === tgtId) break;
    const nbs = dirAdj[u] || [];
    for (const {{ link, to }} of nbs) {{
      if (to in prev) continue;
      prev[to] = {{ link, from: u }};
      if (to === tgtId) {{ head = q.length; break; }}
      q.push(to);
    }}
  }}
  if (!(tgtId in prev)) return null;
  const out = [];
  let cur = tgtId;
  while (prev[cur] && prev[cur].link) {{
    out.unshift(prev[cur].link);
    cur = prev[cur].from;
  }}
  return out;
}}

// Propagate a sign product along the path.
// Conventions: `+` identity · `-` flips · `0` annihilates · `±` marks
// ambiguous carry-through · `unknown` poisons the whole product.
// SYNC: mirrors `ontograph.paths.sign_product` (see tests/test_paths.py).
// If you change one, change both — or move the algebra to a shared layer.
function computeSignProduct(signs) {{
  let current = '+';
  let ambiguous = false;
  for (const s of signs) {{
    if (s === 'unknown' || !s) return 'unknown';
    if (s === '0') return '0';
    if (s === '±') ambiguous = true;  // product carries but becomes ambiguous
    else if (s === '-') current = (current === '+') ? '-' : '+';
    // '+' leaves current unchanged
  }}
  return ambiguous ? '±' : current;
}}

const pathInfoEl = document.getElementById('pathInfo');

function setPathSource(id) {{
  pathState.source = id;
  document.getElementById('pathSrcLbl').textContent = id || '—';
  recomputePath();
}}
function setPathTarget(id) {{
  pathState.target = id;
  document.getElementById('pathTgtLbl').textContent = id || '—';
  recomputePath();
}}
function clearPath() {{
  pathState = {{ source: null, target: null, links: [] }};
  pathLinks.clear();
  pathNodeIds.clear();
  document.getElementById('pathSrcLbl').textContent = '—';
  document.getElementById('pathTgtLbl').textContent = '—';
  pathInfoEl.style.display = 'none';
  refreshHighlight();
}}
function recomputePath() {{
  pathLinks.clear();
  pathNodeIds.clear();
  pathState.links = [];
  if (pathState.source || pathState.target) clearCycleHighlight();  // mutually exclusive
  if (!pathState.source || !pathState.target) {{
    pathInfoEl.style.display = 'none';
    refreshHighlight();
    return;
  }}
  const p = bfsPath(pathState.source, pathState.target);
  if (p === null) {{
    // No path — still show the panel with a no-path message.
    renderPathPanel(null);
    refreshHighlight();
    return;
  }}
  pathState.links = p;
  p.forEach(l => pathLinks.add(l));
  pathNodeIds.add(pathState.source);
  p.forEach(l => pathNodeIds.add(l.target.id || l.target));
  renderPathPanel(p);
  refreshHighlight();
}}
function renderPathPanel(linksOrNull) {{
  const headHtml =
    `<div class="phead">` +
      `<span><b style="color:#fff">${{escapeHtml(pathState.source)}}</b>` +
      ` <span style="color:#888">⇒</span> ` +
      `<b style="color:#fff">${{escapeHtml(pathState.target)}}</b></span>` +
      `<button class="close" onclick="clearPath()">clear</button>` +
    `</div>`;
  if (linksOrNull === null) {{
    pathInfoEl.innerHTML = headHtml +
      `<div class="nopath">No directed path found.</div>`;
    pathInfoEl.style.display = 'block';
    return;
  }}
  if (linksOrNull.length === 0) {{
    pathInfoEl.innerHTML = headHtml +
      `<div class="nopath">Source and target are the same node.</div>`;
    pathInfoEl.style.display = 'block';
    return;
  }}
  const signs = linksOrNull.map(l => l.sign || 'unknown');
  const product = computeSignProduct(signs);
  const pColor =
    product === '+'  ? '#3be083' :
    product === '-'  ? '#ff6a6a' :
    product === '±'  ? '#ffd27a' :
    product === '0'  ? '#9ecbff' :
                       '#c9a0ff';  // unknown
  // Chain: n0 —(sign)→ n1 —(sign)→ n2 ...
  const parts = [`<span class="pnode">${{escapeHtml(pathState.source)}}</span>`];
  linksOrNull.forEach(l => {{
    const s = l.sign || 'unknown';
    parts.push(
      ` <span class="pedge" style="color:${{l.color}}">—${{escapeHtml(s)}}→</span> ` +
      `<span class="pnode">${{escapeHtml(l.target.id || l.target)}}</span>`
    );
  }});
  pathInfoEl.innerHTML = headHtml +
    `<div class="pchain">${{parts.join('')}}</div>` +
    `<div class="pprod">net sign = ` +
      `<span style="color:${{pColor}};font-weight:600">${{escapeHtml(product)}}</span>` +
      ` · <span style="color:#888">${{linksOrNull.length}} edge${{linksOrNull.length === 1 ? '' : 's'}}</span>` +
      `</div>`;
  pathInfoEl.style.display = 'block';
}}

// Wire the path buttons and keep them enabled-state in sync with selection.
const btnUseSrc = document.getElementById('btnUseSrc');
const btnUseTgt = document.getElementById('btnUseTgt');
document.getElementById('btnPathClear').addEventListener('click', clearPath);
btnUseSrc.addEventListener('click', () => {{ if (selectedId) setPathSource(selectedId); }});
btnUseTgt.addEventListener('click', () => {{ if (selectedId) setPathTarget(selectedId); }});
function syncPathButtons() {{
  btnUseSrc.disabled = !selectedId;
  btnUseTgt.disabled = !selectedId;
}}
syncPathButtons();

// ---- Cycle explorer ---------------------------------------------------
const cycleBoxEl = document.getElementById('cycleBox');
const cycleCountEl = document.getElementById('cycleCount');

function renderCycleList() {{
  const cs = DATA.cycles || [];
  cycleCountEl.textContent = `(${{cs.length}})`;
  if (!cs.length) {{
    cycleBoxEl.innerHTML = `<div class="empty">no cycles detected</div>`;
    return;
  }}
  const rows = cs.map((c, i) => {{
    const lagTag = c.lag === 'within_step' ? 'w' : 'a';
    const prod = c.product || 'unknown';
    const prodColor =
      prod === '+' ? '#3be083' :
      prod === '-' ? '#ff6a6a' :
      prod === '±' ? '#ffd27a' :
      prod === '0' ? '#9ecbff' :
                     '#c9a0ff';
    const chain = c.nodes.concat([c.nodes[0]]).join(' → ');
    return `<div class="cyc" data-idx="${{i}}" title="${{escapeHtml(chain)}}">` +
      `<span class="lag ${{lagTag}}" title="${{c.lag}}">${{lagTag}}</span>` +
      `<span class="prod" style="color:${{prodColor}}">${{escapeHtml(prod)}}</span>` +
      `<span class="chain">${{escapeHtml(chain)}}</span>` +
    `</div>`;
  }}).join('');
  cycleBoxEl.innerHTML = rows;
  cycleBoxEl.querySelectorAll('.cyc').forEach(row => {{
    row.addEventListener('click', () => {{
      const idx = parseInt(row.getAttribute('data-idx'), 10);
      selectCycle(idx);
    }});
  }});
}}

function selectCycle(idx) {{
  const cs = DATA.cycles || [];
  if (!cs[idx]) return;
  // Mutually exclusive with path-highlight.
  pathLinks.clear();
  pathNodeIds.clear();
  pathState = {{ source: null, target: null, links: [] }};
  document.getElementById('pathSrcLbl').textContent = '—';
  document.getElementById('pathTgtLbl').textContent = '—';
  pathInfoEl.style.display = 'none';

  cycleLinks.clear();
  cycleNodeIds.clear();
  activeCycleIdx = idx;
  const c = cs[idx];
  c.edge_idxs.forEach(i => {{ if (DATA.links[i]) cycleLinks.add(DATA.links[i]); }});
  c.nodes.forEach(n => cycleNodeIds.add(n));

  cycleBoxEl.querySelectorAll('.cyc.active').forEach(el => el.classList.remove('active'));
  const row = cycleBoxEl.querySelector(`.cyc[data-idx='${{idx}}']`);
  if (row) row.classList.add('active');

  // Snap camera near the first node of the cycle.
  const firstNode = nodeById[c.nodes[0]];
  if (firstNode && typeof firstNode.x === 'number') focusCamera(firstNode);

  refreshHighlight();
}}

function clearCycleHighlight() {{
  if (!cycleLinks.size) return;
  cycleLinks.clear();
  cycleNodeIds.clear();
  activeCycleIdx = null;
  cycleBoxEl.querySelectorAll('.cyc.active').forEach(el => el.classList.remove('active'));
  refreshHighlight();
}}

renderCycleList();

const encycList = document.getElementById('encycList');
const encycCountEl = document.getElementById('encycCount');
encycCountEl.textContent = `(${{DATA.nodes.length}})`;

const sorted = [...DATA.nodes].sort((a, b) =>
  a.id.toLowerCase().localeCompare(b.id.toLowerCase()));

function entryHtml(n) {{
  return `<div class='e' data-id='${{encodeURIComponent(n.id)}}' title='${{escapeHtml(n.type)}}'>` +
         `<span class='dot' style='background:${{n.color}}'></span>` +
         `<span class='n'>${{escapeHtml(n.id)}}</span>` +
         `<span class='t'>${{escapeHtml(n.type)}}</span>` +
         `</div>`;
}}

function renderAlphabetical() {{
  let currentLetter = null;
  const frag = [];
  sorted.forEach(n => {{
    const letter = (n.id[0] || '?').toUpperCase();
    const letterKey = /[A-Z]/.test(letter) ? letter : '#';
    if (letterKey !== currentLetter) {{
      frag.push(`<div class='letter'>${{letterKey}}</div>`);
      currentLetter = letterKey;
    }}
    frag.push(entryHtml(n));
  }});
  encycList.innerHTML = frag.join('');
  encycCountEl.textContent = `(${{sorted.length}})`;
  wireRowClicks();
  reapplyActive();
}}

function renderFiltered(q) {{
  const hits = sorted.filter(n =>
    n.id.toLowerCase().includes(q) || (n.type||'').toLowerCase().includes(q));
  encycList.innerHTML = hits.length
    ? hits.map(entryHtml).join('')
    : `<div style='padding:18px 6px;color:#6a90b5;font-style:italic'>no matches</div>`;
  encycCountEl.textContent = `(${{hits.length}} / ${{sorted.length}})`;
  wireRowClicks();
  const first = encycList.querySelector('.e');
  encycList.querySelectorAll('.e.active').forEach(el => el.classList.remove('active'));
  if (first) first.classList.add('active');
}}

function wireRowClicks() {{
  encycList.querySelectorAll('.e').forEach(el => {{
    el.addEventListener('click', () => {{
      selectNode(decodeURIComponent(el.getAttribute('data-id')));
    }});
  }});
}}

function reapplyActive() {{
  if (!selectedId) return;
  const row = encycList.querySelector(`.e[data-id='${{CSS.escape(encodeURIComponent(selectedId))}}']`);
  if (row) row.classList.add('active');
}}

renderAlphabetical();

const input = document.getElementById('searchInput');

function runSearch() {{
  const q = input.value.trim().toLowerCase();
  if (!q) renderAlphabetical();
  else    renderFiltered(q);
}}

function moveActive(delta) {{
  const rows = [...encycList.querySelectorAll('.e')];
  if (!rows.length) return;
  let idx = rows.findIndex(r => r.classList.contains('active'));
  if (idx < 0) idx = delta > 0 ? -1 : rows.length;
  idx = Math.max(0, Math.min(rows.length - 1, idx + delta));
  rows.forEach(r => r.classList.remove('active'));
  rows[idx].classList.add('active');
  rows[idx].scrollIntoView({{ block: 'nearest' }});
}}

input.addEventListener('input', runSearch);
input.addEventListener('keydown', ev => {{
  if (ev.key === 'ArrowDown')     {{ moveActive(+1); ev.preventDefault(); }}
  else if (ev.key === 'ArrowUp')  {{ moveActive(-1); ev.preventDefault(); }}
  else if (ev.key === 'Enter') {{
    const active = encycList.querySelector('.e.active');
    if (active) {{
      selectNode(decodeURIComponent(active.getAttribute('data-id')));
      ev.preventDefault();
    }}
  }}
  else if (ev.key === 'Escape')   {{ input.value=''; runSearch(); clearSelection(); input.blur(); }}
}});

document.getElementById('clearSel').addEventListener('click', () => {{
  clearSelection();
}});

const _origSelectNode = selectNode;
selectNode = function(id) {{
  _origSelectNode(id);
  encycList.querySelectorAll('.e.active').forEach(el => el.classList.remove('active'));
  const row = encycList.querySelector(`.e[data-id='${{CSS.escape(encodeURIComponent(id))}}']`);
  if (row) {{
    row.classList.add('active');
    row.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
  }}
  syncPathButtons();
}};
const _origClearSelection = clearSelection;
clearSelection = function() {{
  _origClearSelection();
  encycList.querySelectorAll('.e.active').forEach(el => el.classList.remove('active'));
  syncPathButtons();
}};

Graph.d3Force('charge').strength(-55);
Graph.d3Force('link').distance(22);

const ORIG = DATA;
// Read the currently-checked values for a toggle class into a Set.
// `data-val` carries the raw string value (which may contain non-id-safe chars).
function checkedVals(cls) {{
  const out = new Set();
  document.querySelectorAll('.' + cls).forEach(cb => {{
    if (cb.checked) out.add(cb.dataset.val);
  }});
  return out;
}}
function applyFilters() {{
  const hideI = document.getElementById('hideIsolated').checked;
  const allowedTypes = checkedVals('tgType');
  const allowedSigns = checkedVals('tgSign');
  const allowedEcs   = checkedVals('tgEc');
  // `tgDiff` rows only exist in diff mode. If absent, treat diff filtering as
  // an identity so diff-unaware KGs behave exactly as before.
  const diffEnabled = document.querySelectorAll('.tgDiff').length > 0;
  const allowedDiff = diffEnabled ? checkedVals('tgDiff') : null;

  // 1. drop nodes whose entity_type or diff_status is toggled off.
  let nodes = ORIG.nodes.filter(n => {{
    if (!allowedTypes.has(n.type)) return false;
    if (diffEnabled && !allowedDiff.has(n.diff_status || 'unchanged')) return false;
    return true;
  }});
  const allowedIds = new Set(nodes.map(n => n.id));

  // 2. drop edges whose sign/edge_class/diff_status is off, OR whose
  //    endpoints were dropped in step 1.
  let links = ORIG.links.filter(l => {{
    if (!allowedSigns.has(l.sign || 'unknown')) return false;
    if (!allowedEcs.has(l.ec || 'unknown')) return false;
    if (diffEnabled && !allowedDiff.has(l.diff_status || 'unchanged')) return false;
    const sId = (l.source && l.source.id) || l.source;
    const tId = (l.target && l.target.id) || l.target;
    return allowedIds.has(sId) && allowedIds.has(tId);
  }});

  // 3. hide orphans (nodes that now have no edges).
  if (hideI) {{
    const keep = new Set();
    links.forEach(l => {{
      keep.add((l.source && l.source.id) || l.source);
      keep.add((l.target && l.target.id) || l.target);
    }});
    nodes = nodes.filter(n => keep.has(n.id));
  }}

  Graph.graphData({{ nodes, links }});
}}
document.querySelectorAll('.tgType, .tgSign, .tgEc, .tgDiff')
  .forEach(cb => cb.addEventListener('change', applyFilters));
document.getElementById('hideIsolated').addEventListener('change', applyFilters);

// ─── Subgraph export (#42) ──────────────────────────────────────────
// "Visible" = whatever is currently in `Graph.graphData()` after the
// filter pipeline above. The export uses `origIdx` on each link to
// reach back into the untransformed `src_relations`, so the JSON
// export round-trips through `build_3d_html` losslessly.
function downloadBlob(name, mime, body) {{
  const blob = new Blob([body], {{ type: mime }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}}
function visibleSlice() {{
  const {{ nodes, links }} = Graph.graphData();
  const nodeIds = new Set(nodes.map(n => n.id));
  // Filter src_entities by visible node ids, preserving the dict shape.
  const ents = {{}};
  for (const [k, v] of Object.entries(ORIG.src_entities || {{}})) {{
    if (nodeIds.has(k)) ents[k] = v;
  }}
  // Filter src_relations by origIdx so we keep ALL fields (provenance,
  // confidence, conditional_on, etc.) — not just the viewer-flat subset.
  const visIdx = new Set();
  links.forEach(l => {{ if (l.origIdx !== undefined) visIdx.add(l.origIdx); }});
  const rels = (ORIG.src_relations || []).filter((_, i) => visIdx.has(i));
  return {{ entities: ents, relations: rels, n_nodes: nodes.length, n_edges: links.length }};
}}
function refreshExportCount() {{
  const {{ nodes, links }} = Graph.graphData();
  const el = document.getElementById('exportCt');
  if (el) el.textContent = nodes.length + 'n / ' + links.length + 'e';
}}
function exportJSON() {{
  const slice = visibleSlice();
  const out = {{ entities: slice.entities, relations: slice.relations }};
  const stamp = new Date().toISOString().slice(0,19).replace(/[:T]/g, '-');
  downloadBlob(`{title}-subgraph-${{stamp}}.json`,
               'application/json',
               JSON.stringify(out, null, 2));
}}
function exportCSV() {{
  // CSV: source,target,sign,lag,edge_class,relation_type
  const slice = visibleSlice();
  const esc = v => {{
    const s = String(v ?? '');
    return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }};
  const head = 'source,target,sign,lag,edge_class,relation_type\\n';
  const rows = slice.relations.map(r => [
    r.source, r.target, r.sign || 'unknown', r.lag || 'unknown',
    r.edge_class || 'unknown', r.relation_type || ''
  ].map(esc).join(',')).join('\\n');
  const stamp = new Date().toISOString().slice(0,19).replace(/[:T]/g, '-');
  downloadBlob(`{title}-edges-${{stamp}}.csv`, 'text/csv', head + rows + '\\n');
}}
document.getElementById('btnExportJSON').addEventListener('click', exportJSON);
document.getElementById('btnExportCSV').addEventListener('click', exportCSV);
// Independent listener so the export-count badge stays in sync with the
// filter pipeline. Both listeners fire on `change`; ordering is irrelevant
// because `refreshExportCount` only reads `Graph.graphData()`.
document.querySelectorAll('.tgType, .tgSign, .tgEc, .tgDiff, #hideIsolated')
  .forEach(cb => cb.addEventListener('change', refreshExportCount));
refreshExportCount();
}} // end go()

function waitForLib(tries) {{
  if (typeof ForceGraph3D !== 'undefined') {{ go(); return; }}
  if (tries <= 0) {{ go(); return; }}
  setTimeout(() => waitForLib(tries - 1), 100);
}}
waitForLib(40);
</script>
</body></html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
