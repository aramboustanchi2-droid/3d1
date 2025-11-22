# KURDO AI System Audit (2025-11-21)

## 1. Scope

Comprehensive audit of demo versus realistic subsystems after transformation sprint. Focus: Design & Build tab, feasibility logic, architectural intelligence, language depth, and pipeline authenticity.

## 2. Replaced / Upgraded Demo Components

| Area | Previous State (Demo) | New State (Realistic) | Notes |
|------|-----------------------|-----------------------|-------|
| 3D Massing Preview | Random point cloud (Mesh3d alphahull=0) | Deterministic prism mesh via `build_prism_mesh` + `optimize_vertices` | Base polygon inferred from user text (dimensions, shape hint) |
| Geometry Context → Brain | Not passed | `context_data` includes `site_area`, `dimensions`, `massing_shape`, `proposed_height` | Enables auto feasibility trigger |
| Feasibility Pipeline Trigger | Text keyword only ("feasibility"/"massing") | Also auto-trigger if geometric context detected | Robust activation |
| Feasibility Metrics | Static heuristic list with % strings | Computed: footprint_area, floors, height, volume, slenderness, estimated GFA, efficiency, daylight, structural risk | Deterministic and reproducible |
| Style Selection | Random pick from list | Deterministic inference from keywords + bilingual style DB | Uses `style_descriptions.py` |
| Style Knowledge | None | Added climate + structural synergy metadata | Persian + English text |
| Specs Metrics | Random percentages (cost efficiency, AI confidence) | Real geometry & feasibility-derived metrics | Removed fake compute stats |
| Chat Knowledge (Architecture) | Generic activation/status answers | Added Brutalist / Parametric / Organic + structural load distinctions (dead vs live) | Persian & English expansions |
| Council Verdict Translation | Basic | Preserved; unaffected but stable | Ready for deeper semantic injection |
| Evolution Chart | Vega-Lite infinite extent warning | Fixed by explicit index and float lists | No warnings |
| State Persistence | Datetime serialization crash in councils state | ISO8601 conversion in `save_all_states` | Stable persistence |

## 3. Newly Added File

`style_descriptions.py` – Architectural style knowledge base with bilingual summaries, climate considerations, structural synergy hints.

## 4. System Behavior Improvements

- Determinism: Same textual request now yields identical massing + metrics (no RNG leakage).
- Explainability: Feasibility report surfaces intermediate assumptions (floors via avg 3.2 m floor height, slenderness ratio logic).
- Extensibility: Central geometry context pattern allows future integration of actual DXF polygon extraction or image-derived site footprint.
- Language Depth: Technical Persian responses reduce perceived "demo/fake" feel, anchoring system in real architectural vocabulary.

## 5. Remaining Placeholder / Future Enhancements

| Subsystem | Current Limitation | Recommended Upgrade Path |
|-----------|--------------------|---------------------------|
| Council Reasoning Chain | Template proposals & static confidences | Integrate LLM or rule engine to generate dynamic rationale & confidence sourced from feasibility metrics |
| DXF/DWG Input Geometry | Not parsed for extrusion in Design tab | Use `ezdxf` + entity filtering (closed LWPOLYLINE) → convert to prisms with actual heights; map layers to usage types |
| Hard-Shape Detection | Implemented in `mesh_utils.detect_polygon_issues` but not invoked | Integrate pre-extrusion validation to skip/report problematic polygons (self-intersection etc.) |
| Deep Learning Module | Placeholder predictions | Persist actual model artifacts; feed feasibility history for learned gfa optimization |
| Simulation Engine | Returns canned results | Bridge to real external APIs/tools (Ladybug Tools, OpenFOAM CLI, structural FEM library) with async job handling |
| RLHF Module | Weight adjustments ephemeral | Persist weights, add aggregation, allow differential updates by category (architecture / structural) |
| Agent Army Metrics | Simulated counts & latencies | Base metrics on real async tasks, queue depths, average processing times |
| Governance & Directives | Static rule enforcement listing | Add runtime hooks to block disallowed operations and log infractions with severity |
| Multi-Language Translation | Keyword-based substitution | Integrate true translation model; maintain glossary for technical terms to avoid drift |
| Singularity Events | Logging only (thematic) | Optionally tie events to toggling feature flags (e.g., enabling new optimization passes) |

## 6. Edge Cases & Testing Considerations

| Case | Handling | Next Step |
|------|----------|-----------|
| Single dimension only (e.g., "Design a 50m tower") | Creates square footprint (50x50) | Allow diameter inference when "tower" + one number → circular base |
| Non-numeric descriptor ("small museum") | Falls back to default 40x40, height 30 | Add semantic size mapping (small/medium/large) |
| Circle request with both dims ("30x30 circle pavilion") | Treated as circle using first dim radius logic (w/2) | Validate consistent diameter usage (maybe min of both) |
| Very tall dimension causing extreme slenderness | Structural risk escalates to High | Add advisory text recommending lateral system (outrigger / bracing) |
| Missing feasibility trigger words | Auto-trigger due to geometric context | Optionally show badge "Auto Feasibility" |
| Non-English dimension patterns (Persian digits) | Not yet parsed | Add regex for Arabic/Persian numerals |

## 7. Performance & Determinism Notes

- Mesh generation O(n) for vertices + faces; current polygon samples small (<32 points). Scaling: pre-check for >1e4 vertices to enable simplification.
- Deterministic seeding removed: no reliance on `random.seed` for specs; geometry derived solely from parsed values.

## 8. Security / Safety Considerations

- Input parsing regex could be expanded to prevent injection (currently benign). Sanitize future file paths before external tool calls.
- Governance freeze currently UI-only; should implement hard guard (e.g., global flag check inside `brain.process_request`).

## 9. Suggested Roadmap (Prioritized)

1. DXF Footprint Extraction → Replace dimension inference when file uploaded.
2. Hard Shape Validation → Pre-feasibility polygon screening with report export.
3. Council Dynamic Reasoning → Use either small local LLM or pattern-based inference to produce rationale paragraphs.
4. Real Simulation Hooks → Async job manager + progress polling.
5. Enhanced Translation Layer → Formal bilingual glossary & fallback model.
6. Structural Advisory Module → Add lateral system suggestions based on slenderness, height categories.
7. Persistent RLHF → Serialize weight evolution and expose a trend chart.

## 10. Bilingual Summary (FA)

- سیستم از حالت نمایشی خارج شد: تولید جرم سه‌بعدی، محاسبات مساحت، طبقات، حجم و نسبت باریک‌بودن اکنون واقعی و قابل توضیح است.
- گزارش امکان‌سنجی شامل کارایی، نور روز و ریسک سازه‌ای به صورت داده‌محور تولید می‌شود.
- پایگاه سبک‌های معماری با توضیحات اقلیمی و سازه‌ای افزوده شده است.
- موارد باقی‌مانده: منطق واقعی شوراها، تحلیل هندسه DXF، اتصال موتورهای شبیه‌سازی، تقویت ترجمه و یادگیری تقویتی.

## 11. Conclusion

Demo artifacts have been systematically replaced with computational geometry, deterministic feasibility analytics, and architectural knowledge infusion. Remaining placeholders are well-defined with clear upgrade paths. System now provides materially grounded outputs instead of synthetic placeholders.

---
Generated by Audit Task 8 automation.
