# ComfyUI-ImageIR-PromptEngine

**Image IR Prompt Engine** for ComfyUI — write down what an image actually shows, then generate prompts that contain nothing else. Every clause is traced back to a recorded fact, and anything that cannot be traced is removed before the prompt leaves the graph.

**[한국어](#-한국어) · [English](#-english)**

---

# 🇰🇷 한국어

## 소개

이미지를 참조해 프롬프트를 쓰면, 거의 항상 **이미지에 없는 것이 섞여 들어갑니다.** 재질이 흐릿해서 확실하지 않았는데 "새틴"이라고 단정하고, 정면을 보고 있는데 "고개를 돌려 카메라를 본다"처럼 없던 **이전 상태**를 지어내고, 사진은 화면 기준 왼쪽만 알려 주는데 "그녀의 왼손"이라고 **인물 기준**으로 바꿔 씁니다. 결과물은 그럴듯하지만 더 이상 그 이미지가 아닙니다.

이 노드는 그 틈을 구조로 막습니다. 먼저 이미지에서 읽은 사실을 **IMAGE_IR**이라는 문서에 적고, 프롬프트는 **오직 그 문서에서만** 만들어집니다. 마지막에는 완성된 문장을 IR과 한 절씩 대조해서 근거가 없는 부분을 잘라 냅니다.

## 핵심 개념

### IMAGE_IR — 유일한 시각적 출처

속성마다 **확신 수준**을 함께 적습니다. 이 세 가지가 전부입니다.

| 표기 | 뜻 | 프롬프트에 나가는 말 |
|---|---|---|
| `=` | **관찰됨** — 이미지에서 읽음 | 적은 값 **그대로** |
| `~` | **불확실** — 있긴 한데 무엇인지 못 읽음 | **완곡어(hedge)만**. 후보는 절대 안 나감 |
| `!` | **없음** — 찾아봤는데 이미지에 없음 | 아무것도 안 나감 |

```
@frame viewer
@laterality unconfirmed

subject.identity      = a woman
subject.pose          = seated, torso upright
subject.gaze          = toward camera
subject.hands         = resting on the lap
wardrobe.top          = blouse
wardrobe.top_material ~ smooth | satin, silk
subject.jewellery     !
```

`wardrobe.top_material ~ smooth | satin, silk` 한 줄이 이렇게 동작합니다.

- 프롬프트에는 **`smooth blouse`** 로 나갑니다.
- **`satin blouse` 는 나가지 않습니다.** `satin`과 `silk`는 "검토했지만 확정하지 못한 후보"로 기록되어 있어서, 가드가 이 단어를 보면 **규칙 3 위반**으로 잡아냅니다.
- 네거티브 프롬프트에는 `satin, silk`가 자동으로 들어갑니다.

### H1 · H2 · H3 — 세 단계 프롬프트

| 단계 | 내용 |
|---|---|
| **H1** | 핵심 — 누가/무엇이, 어떤 자세로, 어디를 보는지 |
| **H2** | 근거 있는 확장 — 의상, 배경, 조명, 카메라 |
| **H3** | 실제로 쓰는 프롬프트 — H1 + H2 + 상태 유지 문구 + 규칙 6이 허용한 미세 동작 + 비시각적 지시(길이·비율·스타일) |

## 적용되는 9가지 근거 규칙

1. IMAGE_IR에 없는 시각 정보를 **새로 만들지 않습니다.**
2. 관찰된 속성을 **바꾸지 않습니다.**
3. `uncertain` 속성을 **특정 값으로 확정하지 않습니다.**
4. 인물 좌우가 명시적으로 확인되지 않았다면, 화면 기준 좌우를 **인물 기준으로 바꾸지 않습니다.**
5. 동작의 **이전 상태를 지어내지 않습니다.**
6. 동작은 IMAGE_IR과 **모순되지 않을 때만** 새로 생성합니다.
7. **불확실함을 그대로 보존합니다.**
8. IMAGE_IR이 화면 기준 좌표를 주면 **화면 기준 표현을 유지합니다.**
9. 완성 전에 H3의 **모든 시각적 진술을 IMAGE_IR과 대조**하고, 추적되지 않는 것은 제거합니다.

### 규칙 5가 실제로 하는 일

IMAGE_IR이 `gaze = toward camera`일 때 —

```
✗ "she turns from looking away toward the camera"
```

시선 값 자체는 맞지만, **"원래 다른 곳을 보고 있었다"** 는 이미지에 없는 이전 상태입니다. 정지 이미지에는 "이전"이 없습니다.

```
✓ "she maintains gentle eye contact with the camera"
✓ "gaze stays toward camera"        ← 엔진이 자동으로 넣는 상태 유지 문구
```

### 규칙 6 — 허용되는 미세 동작 5가지

`blink` · `breathing` · `finger_movement` · `hair_movement` · `weight_shift`

이것만 가능합니다. 시선 방향 변경, 자세 범주 변경, 팔다리를 크게 올리거나 내리기, 새로운 신체 부위 접촉, 다른 장소로 이동, 사물 조작은 모두 거부됩니다.

미세 동작도 두 가지 조건을 통과해야 합니다.

- **모순 없음** — IR에 `eyes closed`가 있으면 `blink`는 거부됩니다.
- **근거 있음** (`require_motion_anchor`, 기본 켜짐) — "머리카락이 흔들린다"는 말은 **머리카락이 있다는 시각적 주장**입니다. IR에 `subject.hair`가 없으면 규칙 6을 이용해 규칙 1을 우회하는 셈이므로 거부합니다. 끄면 규칙 6을 문자 그대로 읽어 모순만 확인합니다.

## 노드

| 노드 | 하는 일 |
|---|---|
| **Image IR (write)** | IMAGE_IR 작성. 줄 단위 문법 또는 JSON. 기존 IR에 이어 붙이기 가능 |
| **Image IR Merge** | 두 IMAGE_IR 병합. 관찰값을 **덮어쓰지 않습니다** |
| **Image IR Prompt Engine (H3)** | H1/H2/H3 + 네거티브 + 추적표 + 자체 감사 결과 생성 |
| **Image IR Grounding Guard** | 규칙 9. **아무** 프롬프트나 IR과 대조해 근거 없는 절을 제거 |
| **Image IR Inspect** | IR 내용, 확신 수준, 사용 가능한 미세 동작 확인 |

체인 끝에는 항상 **Grounding Guard**를 두세요. 엔진은 자기 출력을 스스로 감사하지만, 그 뒤에 손으로 고치거나 즐겨 쓰는 스타일 문자열을 이어 붙인 텍스트에는 그런 보장이 없습니다.

## 출력

| 출력 | 내용 |
|---|---|
| `h3_prompt` | 실제로 쓰는 프롬프트 |
| `h1_core` / `h2_attributes` | 단계별 프롬프트 |
| `negative_prompt` | IR이 배제한 값 + 불확실 후보 + 없는 것 (전부 자동 도출) |
| `trace` | 절마다 어떤 IR 항목에서 왔는지. 쓰이지 않은 항목, 거부된 동작도 표시 |
| `audit` | 규칙 9 감사 결과 — 유지된 절, 제거된 절, 위반한 규칙 원문 |

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssain3d-lgtm/ComfyUI-ImageIR-PromptEngine
```

추가 파이썬 패키지가 **필요 없습니다.** 엔진은 표준 라이브러리만 씁니다.

## 예제 워크플로

- `example_workflows/image-ir-h3-prompt.json` — IMAGE_IR 작성 → H3 프롬프트 → 가드
- `example_workflows/image-ir-audit-existing-prompt.json` — 이미 있는 프롬프트를 감사. 규칙 1·2·3·4·5·6 위반이 한 절씩 잡히는 것을 확인할 수 있습니다

---

# 🇬🇧 English

## Why

Writing a prompt from a reference image almost always smuggles something in. The fabric was too blurry to read, and the prompt says "satin". The subject is looking straight ahead, and the prompt says "she turns from looking away toward the camera" — a past the photograph never had. The photograph knows only which side of the frame a hand is on, and the prompt says "her left hand". The result is plausible and is no longer that image.

This node closes the gap structurally. What the image shows is recorded once, in an **IMAGE_IR** document. Prompts are composed **only** from that document. The finished text is then compared against it clause by clause, and whatever cannot be traced is removed.

## IMAGE_IR — the authoritative visual source

Every attribute carries how well it is known. There are three levels and no others:

| Mark | Meaning | What may be emitted |
|---|---|---|
| `=` | **observed** — read off the image | the recorded value, **verbatim** |
| `~` | **uncertain** — present, unreadable | the **hedge only**; never the candidates |
| `!` | **absent** — looked for, not there | nothing |

```
@frame viewer
@laterality unconfirmed

subject.identity      = a woman
subject.pose          = seated, torso upright
subject.gaze          = toward camera
subject.hands         = resting on the lap
wardrobe.top          = blouse
wardrobe.top_material ~ smooth | satin, silk
subject.jewellery     !
```

That one uncertain line does three things: the prompt says **`smooth blouse`**; a prompt that says `satin blouse` is rejected under rule 3, because `satin` and `silk` are recorded as readings that were weighed and not confirmed; and `satin, silk` land in the negative prompt automatically.

JSON is accepted too, and is what the nodes pass between themselves.

## H1 · H2 · H3

| Tier | Contents |
|---|---|
| **H1** | the core — who or what, how posed, where looking |
| **H2** | the grounded expansion — wardrobe, scene, lighting, camera |
| **H3** | the render-ready prompt — H1 + H2 + continuity wording + whatever micro-motion rule 6 permits + non-visual directives |

## The nine grounding rules

1. Never introduce a visual fact that is not present in IMAGE_IR.
2. Never change an observed attribute.
3. Never resolve an "uncertain" attribute into a specific value.
4. Never convert viewer-left / viewer-right into subject-left / subject-right unless subject laterality is explicitly confirmed.
5. Do not invent an initial state for motion.
6. Motion may be newly generated ONLY when it does not contradict IMAGE_IR.
7. Preserve uncertainty.
8. When IMAGE_IR provides viewer-relative geometry, preserve viewer-relative terminology.
9. Before finalizing, compare every visual statement in the H3 prompt against IMAGE_IR, and remove anything that cannot be traced to it or to an explicitly allowed micro-motion.

### Rule 5 in practice

With `gaze = toward camera` in the IR:

```
✗ "she turns from looking away toward the camera"
```

The gaze value is correct and the sentence is still wrong: *"was looking away"* is a state before the frame, and a still frame has no before.

```
✓ "she maintains gentle eye contact with the camera"
✓ "gaze stays toward camera"        ← the continuity wording the engine adds
```

### Rule 6 — the five micro-motions

`blink` · `breathing` · `finger_movement` · `hair_movement` · `weight_shift`

Everything else is refused: changing gaze direction, changing pose category, raising or lowering a limb substantially, touching a new body part, moving to another location, manipulating an object.

Each of the five still has to clear two conditions:

- **No contradiction** — `eyes closed` in the IR rules out `blink`.
- **An anchor** (`require_motion_anchor`, on by default) — "a few strands of hair drifting" asserts that there *is* hair. With no `subject.hair` in the IR, allowing it would use rule 6 as a side door around rule 1. Turn the toggle off to read rule 6 literally, checking only for contradiction.

## Nodes

| Node | What it does |
|---|---|
| **Image IR (write)** | Author IMAGE_IR, in the line syntax or as JSON; optionally extending an existing document |
| **Image IR Merge** | Combine two documents. Sharpening is allowed; **overwriting an observation is not** |
| **Image IR Prompt Engine (H3)** | Compose H1/H2/H3, a derived negative, a clause trace, and a self-audit |
| **Image IR Grounding Guard** | Rule 9. Audit **any** prompt against the IR and strip what does not trace |
| **Image IR Inspect** | Show the document, its certainty levels, and the micro-motions it can carry |

Chain the **Grounding Guard** last. The engine audits its own output, but text that has been edited by hand or concatenated with a favourite style string carries no such guarantee — and rule 9 asks for the check on the final text.

## Outputs

| Output | Contents |
|---|---|
| `h3_prompt` | the render-ready prompt |
| `h1_core` / `h2_attributes` | the lower tiers, for inspecting a mistake at the altitude it lives at |
| `negative_prompt` | poles the IR ruled out, candidates it refused to resolve, attributes it recorded as absent — all derived, none a taste preference |
| `trace` | every clause with the IR paths that licensed it, plus facts left unused and motions refused |
| `audit` | the rule 9 report: clauses kept, clauses removed, and the text of each rule broken |

`on_violation` decides what a failure does: `filter` removes the offending clauses (default), `error` stops the queue, `report_only` passes the text through and still reports.

## Grounding, concretely

The guard classifies each clause and keeps only three kinds:

- **grounded** — every content word traces to an IR fact, or to a paraphrase of an observed wording listed in a reviewable equivalence table (so "holds eye contact with the camera" traces to `gaze = toward camera`, while "gazing warmly" does not — warmth was never observed).
- **micro-motion** — one of the five, cleared for contradiction and anchor.
- **directive** — render settings and durations, which claim nothing about the picture.

Everything else is a finding that names the rule, the clause, and the fix. A gendered pronoun counts as a claim about the subject: `she` needs the IR to have recorded a subject it fits, which is what `check_pronouns` governs.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssain3d-lgtm/ComfyUI-ImageIR-PromptEngine
```

**No extra Python packages.** The engine is standard library only, which is also why it can be imported and tested without ComfyUI:

```python
from imageir import parse_dsl, compose, audit

ir = parse_dsl("subject.identity = a woman\nsubject.gaze = toward camera")
print(compose(ir, motions=["blink"]).h3)
print(audit(ir, "she turns from looking away toward the camera").report())
```

## Example workflows

- `example_workflows/image-ir-h3-prompt.json` — author IMAGE_IR, compose H3, guard the result
- `example_workflows/image-ir-audit-existing-prompt.json` — audit a prompt you already have; its six clauses break rules 1, 2, 3, 4, 5 and 6, one each

## Development

```bash
ruff check .
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

MIT
