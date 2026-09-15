# H3 USER_INTENT integration (issue #5)

## Contract and source review

Target starts at `5fa8b1a364ad3609576a1dab873e82cc08407849`; PR base is
`claude/confident-babbage-gja9b1`. Reference repository:
https://github.com/ssain3d-lgtm/Comfyui-H3-Prompt-Maker-lgtm/tree/af9d7496440483b171ac24696c642cae922df2cb

Reviewed `h3_prompts.py`, `nodes.py` (Prompt Architect, UI and Instant), and
`server_routes.py` (`prepare_generation` and generation route). Architect builds
mode-specific user content and sends it with a system template; Instant rebuilds
the overlay request and shares the HTTP route's preparation. Both ask a model
for final prompt text. Here the model produces a separate, inspectable intent.

Reuse concepts, not source implementation: exact 3/6-section order, first/final
alignment syntax, one subject with multiple role-bearing assets, positive
retention markers, English authoring with original-language dialogue, and
24fps duration rounded upward to the 17k+5 grid (maximum 362 frames).
Do not import the overlay, HTTP routes, backend clients, CLI runners, model
loading, VRAM management, Fast/Full prompt stacks, or permissive embellishment.
No reference-repository runtime dependency or copied implementation is needed.

## Architecture

`intent.py`: immutable USER_INTENT_v1 and evidence-bearing authored clauses;
structured generation through the existing backend's `generate_text`. Preserve
raw Korean/English inputs and require each translated clause to cite an exact
input span. Separate actions, camera, progression, style, audio, constraints,
dialogue and explicitly described start/end states.

`references.py`: REFERENCE_PACK_v1 with explicit image/video/audio roles and
subject grouping. Image facts retain the original IMAGE_IR object/schema.
Media files stay in the downstream workflow; the pack describes roles only.

`h3_modes.py`: validated AUTO/manual routing, frame-grid timing and deterministic
composition. First/final anchors come exclusively from their respective IRs.
Ref2VA filters facts by asset role; it never merges a motion source's wardrobe.

`provenance.py`: source-labelled clauses and strict document verification.
Reconstruct expected composition rather than accepting caller-supplied source
labels as authority. Reject untraced additions and boundary contradictions;
preserve IMAGE_IR uncertainty, H3 syntax and authored audio. Semantic fidelity
of translation remains an LLM limitation, explicitly exposed in the trace.

Add ComfyUI author/import, reference, router, composer and guard nodes while
preserving legacy nodes and workflows. Add env-token input to existing config.
Python >=3.10, standard-library runtime only. Never lower the 88% coverage gate.

## Execution and validation

- [x] Add failing schema, authoring, role and routing tests; implement immutable
  schemas with JSON roundtrips and existing-backend structured authoring.
- [x] Add five-mode and adversarial provenance tests; implement composition,
  frame timing, role filtering and boundary validation.
- [x] Add ComfyUI graph and env-secret tests; wire nodes, examples and docs.
- [x] Run complete unittest suite, Ruff, coverage >=88%, secret scan and clean
  import. Fix baseline numpy imaging parity failures separately.
- [ ] Review diff, commit logical changes, push specified branch, wait for all
  push CI jobs to pass, open PR against requested base and verify PR status.

No real model/network in automated tests. Manual Windows ComfyUI checks must
cover local/Gemini authoring, Korean semantics, actual reference attachment
order and video rendering. Long multi-render sequences are outside this MVP:
reject durations above 362/24 instead of silently truncating.
