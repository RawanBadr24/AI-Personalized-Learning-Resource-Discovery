"""Turn the pipeline result into the final markdown answer (structure from the design document)."""
from __future__ import annotations


def format_final_response(result: dict) -> str:
    profile, roadmap = result["profile"], result["roadmap"]
    lines = [
        "## 1. Your learning goal",
        f"**Subject:** {profile.get('subject', '-')}  \n**Level:** {profile.get('level', '-')}  \n"
        f"**Objectives:** {'; '.join(profile.get('objectives') or ['-'])}",
        "",
        "## 2. Topics & recommended sequence",
        *[f"{s['stage']}. {s['topic']}" for s in roadmap["stages"]],
        "",
        "## 3. Ranked resources",
    ]
    if not result["ranked"]:
        lines.append("_No resource passed evaluation. Try broadening the request or relaxing preferences._")
    for r in result["ranked"]:
        lines += [
            f"**{r['rank']}. [{r['title']}]({r['url']})** — {r['platform']} · {r.get('price') or 'Unknown'}"
            f" · difficulty: {r.get('difficulty') or 'not available'} · score {r['final_score']:.2f}",
            f"   - Why: {r.get('reason') or 'Matched the requested topics.'}",
        ]
    lines += ["", "## 4. Step-by-step roadmap"]
    for s in roadmap["stages"]:
        lines.append(f"### Stage {s['stage']}: {s['topic']}  _(~{s['estimated_hours']} h, {s['time_basis']})_")
        lines += [f"- 📚 [{r['title']}]({r['url']})" for r in s["resources"]] or [f"- ⚠️ {s['note']}"]
        lines += [f"- ✏️ {t}" for t in s["practice_tasks"]]
    total = f"**Total estimated time:** ~{roadmap['total_estimated_hours']} hours"
    if roadmap.get("estimated_weeks"):
        total += f" (~{roadmap['estimated_weeks']} weeks at your pace)"
    lines += ["", "## 5. Estimated time", total]
    lines += ["", "## Notes & transparency",
              f"- Search/evaluation cycles used: {result['cycles_used']} (max 3); evaluated {result['evaluated']} resources, rejected {result['rejected']}.",
              "- Time estimates use resource durations when available, otherwise a default estimate.",
              "- Web results have unknown price unless stated; check before enrolling."]
    if result["coverage_gaps"]:
        lines.append(f"- ⚠️ No approved resource found for: {', '.join(result['coverage_gaps'])}.")
    return "\n".join(lines)
