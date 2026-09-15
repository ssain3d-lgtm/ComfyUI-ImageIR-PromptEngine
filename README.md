# ComfyUI-ImageIR-PromptEngine

Read images into **IMAGE_IR**, author future video instructions into **USER_INTENT**, and compose five H3 modes with **IMAGE_IR / USER_INTENT / H3_RULE** provenance. Existing image-only workflows remain available.

**[한국어](#-한국어) · [English](#-english)**

```
[Load Image] → [Backend Config] → [Image IR Analyzer] → IMAGE_IR → [MiniMax H3 Composer] → [Grounding Guard]
```

---

# 🇰🇷 한국어

## 왜 필요한가

참조 이미지를 보고 프롬프트를 쓰면 거의 항상 **이미지에 없는 것이 섞여 들어갑니다.** 재질이 흐릿해 확신할 수 없었는데 "새틴"이라 단정하고, 정면을 보고 있는데 "고개를 돌려 카메라를 본다"처럼 없던 **이전 상태**를 지어내고, 사진은 화면 기준 왼쪽만 알려 주는데 "그녀의 왼손"이라고 **인물 기준**으로 바꿔 씁니다. 결과는 그럴듯하지만 더 이상 그 이미지가 아닙니다.

이 확장은 그 틈을 구조로 막습니다. 비전 모델이 읽은 내용을 **IMAGE_IR** 문서에 한 번 기록하고, 프롬프트는 **오직 그 문서에서만** 만들어지며, 완성된 문장은 IR과 한 절씩 대조해 근거 없는 부분을 잘라 냅니다.

## 세 가지 개념 — 헷갈리지 마세요

| | 무엇인가 |
|---|---|
| **IMAGE_IR** | **Intermediate Representation(중간 표현).** 원본 이미지와 프롬프트 생성 사이에 놓이는 구조화된 시각 기술. 이 프로젝트의 유일한 권위 있는 출처 |
| **일반 근거 프롬프트** | 어떤 모델에도 넣을 수 있는 쉼표 구분 프롬프트. 티어는 `CORE` / `DETAIL` / `FINAL` |
| **MiniMax H3 프롬프트** | MiniMax H3 영상 모델의 **공식 I2VA 포맷**. 필드명과 순서가 고정됨 |

> **중요**: 내부 티어는 예전에 H1/H2/H3였지만 지금은 **CORE/DETAIL/FINAL**입니다. `H3`는 오직 MiniMax H3만 가리킵니다.

## 기본 워크플로

```
[Load Image]
      ↓
[Image IR Backend Config]     ← 모델·서버·토큰을 한 번만 설정
      ↓
[Image IR Analyzer]           ← 이미지를 읽어 IMAGE_IR 생성
      ↓
   IMAGE_IR
      ↓
[MiniMax H3 Prompt Composer]  ← 또는 [Image IR Prompt Composer]
      ↓
[Image IR Grounding Guard]    ← prompt_format = minimax_h3
      ↓
   최종 프롬프트
```

llama.cpp를 ComfyUI에서 직접 띄우려면 옆에 **[Image IR Backend Control]** 을 두면 됩니다.

IMAGE_IR을 손으로 쓰는 **[Image IR Write (manual)]**, **[Image IR Merge]**, **[Image IR Inspect]** 는 고급/디버그용으로 그대로 남아 있습니다.

## IMAGE_IR — 확신 수준 + 신뢰도

속성마다 **두 가지 다른 정보**를 함께 기록합니다.

**certainty** — 어떤 종류의 주장인가

| 표기 | 뜻 | 프롬프트에 나가는 말 |
|---|---|---|
| `=` | **observed** — 이미지에서 읽음 | 적은 값 **그대로** |
| `~` | **uncertain** — 있긴 한데 무엇인지 못 읽음 | **완곡어(hedge)만** |
| `!` | **absent** — 찾아봤는데 없음 | 아무것도 안 나감 |

**confidence** — 그 주장을 얼마나 믿을 수 있는가 (0.0–1.0, 선택)

둘은 **서로 다른 질문**이며 어느 쪽도 다른 쪽을 대체하지 않습니다. `observed` + `0.55`(분명히 보이지만 해상도가 나쁨)도, `uncertain` + `0.9`(확실하게 판독 불가)도 가능합니다.

`confidence`가 **없으면 없는 대로 둡니다.** 사람이 손으로 쓴 문서에는 원래 숫자가 없으며, 거기에 `1.0`을 채워 넣는 것은 없는 정밀도를 지어내는 일입니다.

```json
{
  "value": "likely sheer nude tights",
  "certainty": "uncertain",
  "confidence": 0.78,
  "verification_required": true,
  "evidence": "uniform leg tone and subtle surface sheen"
}
```

`verification_required`는 **향후 검증기(verifier) 라우팅을 위한 자리**입니다. 스키마는 준비되어 있지만 검증기 자체는 이번 패스에서 구현하지 않았습니다.

손으로 쓸 때는 줄 문법을 씁니다:

```
@frame viewer
@laterality unconfirmed

subject.identity      = a woman
subject.gaze          = toward camera @0.95
wardrobe.top          = blouse
wardrobe.top_material ~ smooth | satin, silk @0.42 @verify
subject.jewellery     !
```

### 불확실 후보는 네거티브 프롬프트가 **아닙니다**

`material ~ smooth | satin, silk` 는 이런 뜻입니다.

- 프롬프트에는 **`smooth blouse`** 로 나갑니다
- `satin blouse` 를 쓰려 하면 **규칙 3 위반**으로 잡힙니다
- **`satin`, `silk` 는 네거티브 프롬프트에 들어가지 않습니다**

마지막 항목이 핵심 수정입니다. *"새틴인지 실크인지 판독할 수 없었다"* 는 *"새틴이 아니다"* 가 **아닙니다.** 판독 실패를 배제 사실로 바꾸면 이미지가 뒷받침한 적 없는 제약을 지어내는 것이고, 정답일 수도 있는 값에서 생성을 밀어내게 됩니다.

후보는 가드가 **주장을 막기 위해** 계속 보관합니다. 주장하지 않는 것과 부정하는 것은 다릅니다.

네거티브에 들어가는 것은 두 가지뿐입니다: **absent로 기록된 속성**, 그리고 **관찰된 값이 배제하는 반대 극**(시선이 카메라를 향한다면 "looking away"는 배제됨).

## 9가지 근거 규칙

1. IMAGE_IR에 없는 시각 정보를 **새로 만들지 않습니다.**
2. 관찰된 속성을 **바꾸지 않습니다.**
3. `uncertain` 속성을 **특정 값으로 확정하지 않습니다.**
4. 인물 좌우가 명시적으로 확인되지 않았다면, 화면 기준 좌우를 **인물 기준으로 바꾸지 않습니다.**
5. 동작의 **이전 상태를 지어내지 않습니다.**
6. 동작은 IMAGE_IR과 **모순되지 않을 때만** 새로 생성합니다.
7. **불확실함을 그대로 보존합니다.**
8. IMAGE_IR이 화면 기준 좌표를 주면 **화면 기준 표현을 유지합니다.**
9. 완성 전에 **모든 시각적 진술을 IMAGE_IR과 대조**하고, 추적되지 않는 것은 제거합니다.

### 규칙 5가 실제로 하는 일

IMAGE_IR이 `gaze = toward camera`일 때 —

```
✗ "she turns from looking away toward the camera"
✓ "she maintains gentle eye contact with the camera"
✓ "gaze stays toward camera"        ← 엔진이 넣는 상태 유지 문구
```

시선 값 자체는 맞지만 **"원래 다른 곳을 보고 있었다"** 는 이미지에 없는 이전 상태입니다. 정지 이미지에는 "이전"이 없습니다.

### 규칙 6 — 허용되는 미세 동작 5가지

`blink` · `breathing` · `finger_movement` · `hair_movement` · `weight_shift`

시선 방향 변경, 자세 범주 변경, 팔다리를 크게 올리거나 내리기, 새로운 신체 부위 접촉, 이동, 사물 조작은 모두 거부됩니다.

미세 동작도 두 조건을 통과해야 합니다.

- **모순 없음** — IR에 `eyes closed`가 있으면 `blink`는 거부
- **근거 있음** (`require_motion_anchor`, 기본 켜짐) — "머리카락이 흔들린다"는 **머리카락이 있다는 시각적 주장**입니다. IR에 `subject.hair`가 없으면 규칙 6으로 규칙 1을 우회하는 셈이라 거부합니다

## 백엔드 설정

### api_token 과 max_tokens 는 완전히 다릅니다

| 필드 | 뜻 |
|---|---|
| **`api_token`** | **인증** — API 키. Gemini에는 필수, 로컬 서버에는 보통 비워 둠 |
| **`max_tokens`** | **생성 길이 상한** — 응답이 몇 토큰까지 나올 수 있는지. 인증과 무관 |

둘을 섞으면 "모든 요청이 거부되거나 모든 응답이 잘리는" 설정이 만들어집니다.

**직접 입력한 토큰은 workflow/PNG metadata에 저장될 수 있습니다.** `api_token_env`에 환경변수 이름만 입력하고 `api_token`은 비워 두세요. 노드 출력·디버그·backend 예외는 마스킹합니다. llama.cpp 로컬 실행 시 키는 커맨드라인 대신 환경변수로 전달합니다.

### A. llama.cpp — ComfyUI에서 직접 실행

```
provider          = llama_cpp
server_mode       = launch_local
llama_server_path = /opt/llama.cpp/llama-server
gguf_model_path   = /models/gemma-3-12b-it-Q4_K_M.gguf
mmproj_path       = /models/mmproj-gemma-3-12b-f16.gguf   ← 비전 모델에 필수
host              = 127.0.0.1
port              = 8080
context_size      = 16384
gpu_layers        = 999
extra_args        = --jinja
model_name        = gemma-3-12b
```

**[Image IR Backend Control]** 에서 `action = start`. 실행되는 명령은 Backend Config의 `summary` 출력에 그대로 표시됩니다:

```
llama-server --model <gguf> --host 127.0.0.1 --port 8080 \
             --ctx-size 16384 --n-gpu-layers 999 --mmproj <mmproj> --jinja
```

준비물: llama.cpp 빌드(또는 릴리스 바이너리), GGUF 가중치, 비전용 **mmproj** 파일.

안전 장치:

- 해당 포트가 이미 응답하면 **두 번째 서버를 띄우지 않습니다**
- `stop`은 **이 확장이 띄운 프로세스만** 종료합니다. 사용자가 직접 띄운 llama.cpp나 LM Studio는 건드리지 않습니다
- 경로 오류·포트 충돌·기동 실패는 ComfyUI를 죽이지 않고 로그 마지막 줄과 함께 상태로 보고됩니다

### B. llama.cpp — 이미 실행 중

```
provider    = llama_cpp   (또는 openai_compatible — 동일하게 동작)
server_mode = connect_existing
base_url    = http://127.0.0.1:8080
model_name  = gemma-3-12b
api_token   =            ← 보통 비움
```

### C. LM Studio

LM Studio는 **같은 OpenAI 호환 클라이언트**를 씁니다. 별도 구현이 없습니다.

```
provider    = openai_compatible
server_mode = connect_existing
base_url    = http://127.0.0.1:1234
model_name  = google/gemma-3-12b       ← LM Studio가 표시하는 모델 id
api_token   =
```

LM Studio에서 **Local Server를 시작**하고 비전 지원 모델을 로드해 두세요.

### D. Gemini

```
provider   = gemini
model_name = gemini-2.5-flash          ← 모델명은 고정되어 있지 않음
api_token  = <API 키>
max_tokens = 4096
```

추가 파이썬 패키지가 **필요 없습니다.** REST API를 직접 호출하므로 선택적 의존성 누락으로 플러그인 임포트가 깨지는 경로 자체가 없습니다. API 키는 URL이 아니라 `x-goog-api-key` 헤더로 전달됩니다.

## 노드

| 노드 | 하는 일 |
|---|---|
| **Image IR Backend Config** | 공급자·모델·생성 설정·토큰을 한 번에 지정 |
| **Image IR Backend Control** | 로컬 llama-server `start` / `stop` / `restart` / `status` |
| **Image IR Analyzer** | IMAGE → IMAGE_IR. 저신뢰 속성 목록과 디버그 정보도 출력 |
| **MiniMax H3 Prompt Composer** | 공식 I2VA 포맷 H3 프롬프트 생성 |
| **Image IR Prompt Composer** | 일반 근거 프롬프트 (CORE / DETAIL / FINAL) |
| **Image IR Grounding Guard** | 규칙 9. `prompt_format = plain` 또는 `minimax_h3` |
| **Image IR Write (manual)** | IMAGE_IR 직접 작성 (고급/디버그) |
| **Image IR Merge** | 두 IMAGE_IR 병합. 관찰값을 덮어쓰지 않음 |
| **Image IR Inspect** | IR 내용·확신 수준·사용 가능한 미세 동작 확인 |
| **Image IR Extractor Prompt** | 분석 프롬프트 원문 출력 (수정용) |

체인 끝에는 항상 **Grounding Guard**를 두세요. 컴포저는 자기 출력을 스스로 감사하지만, 그 뒤에 손으로 고치거나 스타일 문자열을 이어 붙인 텍스트에는 그런 보장이 없습니다.

## MiniMax H3 포맷

공식 I2VA 포맷을 그대로 따릅니다 — 필드명, 순서, `라벨: 값` 구두점, 블록 사이 빈 줄까지.

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a woman, seated, ...

overall_soundscape: Rain taps the glass; a chair creaks.

non_diegetic_music: N/A
```

- **피사체 동작**과 **카메라 동작**은 별도 필드로 분리됩니다. 카메라 움직임은 문장 끝에 라벨로 붙이지 않고 샷 안에 자연스러운 영어 문장으로 씁니다(가이드 지시)
- 오디오 두 필드는 **저자가 씁니다.** 정지 이미지에는 소리가 없으므로 IR이 근거를 줄 수 없고, 시각 근거 기준으로 감사하지 않습니다. 비우면 `N/A`
- 가드의 `prompt_format = minimax_h3` 는 필드명·마커·오디오 블록을 보존하면서 **시각 기술 부분만** 감사합니다

## 분석 프롬프트

`prompts/image_ir_extractor.txt` — 거대한 파이썬 상수가 아니라 편집 가능한 파일입니다. 피사체·얼굴·머리·상의·하의·색·구조·재질 불확실성·투명도·스타킹 vs 맨다리·신발·손·자세·다리 기하·환경·사물·배경·조명·프레이밍·카메라 각도·공간 관계를 훑고, 알려진 위험 영역(인물 좌우 vs 화면 좌우, 재질 환각, 미묘한 색조, 시스루 스타킹, 배경 재질 오판, 다리 꼬임, 카메라 각도)을 명시적으로 경고합니다.

## 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssain3d-lgtm/ComfyUI-ImageIR-PromptEngine
```

**추가 파이썬 패키지가 필요 없습니다.** 엔진은 표준 라이브러리만 씁니다.

## 예제 워크플로

- `example_workflows/image-ir-analyze-to-h3.json` — 이미지 → 분석 → H3 → 가드 (기본 파이프라인)
- `example_workflows/image-ir-manual-prompt.json` — IMAGE_IR 직접 작성 → 일반 프롬프트 → 가드
- `example_workflows/image-ir-audit-existing-prompt.json` — 이미 있는 프롬프트 감사

---

# 🇬🇧 English

## Why

Writing a prompt from a reference image almost always smuggles something in. The fabric was too blurry to read, and the prompt says "satin". The subject is looking straight ahead, and the prompt says "she turns from looking away toward the camera" — a past the photograph never had. The photograph knows only which side of the frame a hand is on, and the prompt says "her left hand". The result is plausible and is no longer that image.

This extension closes the gap structurally. A vision model's reading is recorded once, in an **IMAGE_IR** document; prompts are composed **only** from it; and the finished text is compared against it clause by clause, with whatever cannot be traced removed.

## Three things, kept apart

| | What it is |
|---|---|
| **IMAGE_IR** | **Intermediate Representation** — a structured visual description that sits between the raw image and prompt generation. The one authoritative source in this project |
| **Generic grounded prompt** | A comma-separated prompt for any model. Tiers are `CORE` / `DETAIL` / `FINAL` |
| **MiniMax H3 prompt** | The **official I2VA format** of the MiniMax H3 video model, with fixed field names and order |

> **Note**: the internal tiers used to be called H1/H2/H3. They are now **CORE/DETAIL/FINAL**, and `H3` refers only ever to MiniMax H3. A field named `h3_prompt` that was not an H3 prompt was a trap.

## The default workflow

```
[Load Image] → [Image IR Backend Config] → [Image IR Analyzer] → IMAGE_IR
             → [MiniMax H3 Prompt Composer] → [Image IR Grounding Guard]
```

Put **[Image IR Backend Control]** beside it to start llama.cpp from ComfyUI. Manual **Write / Merge / Inspect** remain as advanced and debugging tools.

## IMAGE_IR — certainty *and* confidence

Each attribute carries two different pieces of information.

**certainty** — what kind of claim this is:

| Mark | Meaning | What may be emitted |
|---|---|---|
| `=` | **observed** — read off the image | the recorded value, **verbatim** |
| `~` | **uncertain** — present, unreadable | the **hedge only** |
| `!` | **absent** — looked for, not there | nothing |

**confidence** — how much weight to put on that claim, 0.0–1.0, optional.

They answer different questions and neither replaces the other: an attribute can be observed at 0.55 (clearly present, poorly resolved) or uncertain at 0.9 (confidently unresolvable).

An **unstated** confidence stays unstated. A hand-written document never had a number, and filling in 1.0 would be fabricated precision.

```json
{
  "value": "likely sheer nude tights",
  "certainty": "uncertain",
  "confidence": 0.78,
  "verification_required": true,
  "evidence": "uniform leg tone and subtle surface sheen"
}
```

`verification_required` is the seat for future verifier routing. The schema supports it; the verifier itself is deliberately not built yet.

The hand-authoring syntax:

```
@frame viewer
@laterality unconfirmed

subject.identity      = a woman
subject.gaze          = toward camera @0.95
wardrobe.top          = blouse
wardrobe.top_material ~ smooth | satin, silk @0.42 @verify
subject.jewellery     !
```

### Uncertain candidates are **not** negative prompts

`material ~ smooth | satin, silk` does three things:

- the prompt says **`smooth blouse`**
- a prompt that says `satin blouse` is rejected under rule 3
- **`satin` and `silk` do NOT enter the negative prompt**

That last point is the correction that matters. *"We could not tell whether it is satin"* is not *"it is not satin"*. Turning a failed reading into an exclusion invents a constraint the image never supported, and steers generation away from what may well be the right answer.

The candidates are still kept, for the guard to stop a prompt from *asserting* one. Refusing to assert is not the same as denying.

Only two things become negatives: attributes recorded **absent**, and the poles an observed value **rules out** (gaze toward camera excludes "looking away").

## The nine grounding rules

1. Never introduce a visual fact that is not present in IMAGE_IR.
2. Never change an observed attribute.
3. Never resolve an "uncertain" attribute into a specific value.
4. Never convert viewer-left / viewer-right into subject-left / subject-right unless subject laterality is explicitly confirmed.
5. Do not invent an initial state for motion.
6. Motion may be newly generated ONLY when it does not contradict IMAGE_IR.
7. Preserve uncertainty.
8. When IMAGE_IR provides viewer-relative geometry, preserve viewer-relative terminology.
9. Before finalizing, compare every visual statement against IMAGE_IR and remove what cannot be traced.

### Rule 5 in practice

With `gaze = toward camera` in the IR:

```
✗ "she turns from looking away toward the camera"
✓ "she maintains gentle eye contact with the camera"
✓ "gaze stays toward camera"        ← the continuity wording the composer adds
```

The gaze value is right and the sentence is still wrong: *"was looking away"* is a state before the frame, and a still frame has no before.

### Rule 6 — the five micro-motions

`blink` · `breathing` · `finger_movement` · `hair_movement` · `weight_shift`

Everything else is refused: changing gaze direction, changing pose category, raising or lowering a limb, touching a new body part, moving to another location, manipulating an object.

Each still clears two conditions — **no contradiction** (`eyes closed` rules out `blink`) and **an anchor** (`require_motion_anchor`, on by default: "a few strands of hair drifting" asserts that there *is* hair, so with no `subject.hair` recorded it would use rule 6 as a side door around rule 1).

## Backends

### `api_token` and `max_tokens` are different things

| Field | Meaning |
|---|---|
| **`api_token`** | **Authentication** — the API key. Required for Gemini, usually empty for a local server |
| **`max_tokens`** | **Reply length cap** — how many tokens the answer may run to. Nothing to do with authentication |

Collapsing them produces a config that either rejects every request or truncates every answer.

**Direct api_token widgets ARE saved in workflows and PNG metadata.** Prefer `api_token_env` (an environment variable name), leaving `api_token` blank. Node outputs and backend errors are masked. For a locally launched llama.cpp it travels in the **environment**, not on the command line, so it is not readable from `ps`.

### A. llama.cpp, launched from ComfyUI

```
provider          = llama_cpp
server_mode       = launch_local
llama_server_path = /opt/llama.cpp/llama-server
gguf_model_path   = /models/gemma-3-12b-it-Q4_K_M.gguf
mmproj_path       = /models/mmproj-gemma-3-12b-f16.gguf   ← required for vision
host              = 127.0.0.1
port              = 8080
context_size      = 16384
gpu_layers        = 999
extra_args        = --jinja
model_name        = gemma-3-12b
```

Then **[Image IR Backend Control]** with `action = start`. The exact command is printed in Backend Config's `summary`:

```
llama-server --model <gguf> --host 127.0.0.1 --port 8080 \
             --ctx-size 16384 --n-gpu-layers 999 --mmproj <mmproj> --jinja
```

You need: a llama.cpp build (or release binary), GGUF weights, and the **mmproj** projector file for vision.

Safeguards: it never starts a second server on a port that already answers; `stop` only ever stops a process **this extension started**, leaving your own llama.cpp or LM Studio alone; and a bad path, a port conflict or a failed startup is reported as a status with the last lines of server output rather than taking ComfyUI down.

### B. llama.cpp, already running

```
provider    = llama_cpp        (or openai_compatible — identical behaviour)
server_mode = connect_existing
base_url    = http://127.0.0.1:8080
model_name  = gemma-3-12b
api_token   =                  ← usually empty
```

### C. LM Studio

LM Studio uses the **same OpenAI-compatible client**. There is no second implementation.

```
provider    = openai_compatible
server_mode = connect_existing
base_url    = http://127.0.0.1:1234
model_name  = google/gemma-3-12b     ← the model id LM Studio shows
api_token   =
```

Start LM Studio's **Local Server** with a vision-capable model loaded.

### D. Gemini

```
provider   = gemini
model_name = gemini-2.5-flash        ← never hard-coded; names turn over
api_token  = <API key>
max_tokens = 4096
```

**No extra Python package.** The REST API is called directly, so there is no optional dependency whose absence could break the plugin's import. The key goes in the `x-goog-api-key` header, not the URL.

## Nodes

| Node | What it does |
|---|---|
| **Image IR Backend Config** | Provider, model, generation settings and token, chosen once |
| **Image IR Backend Control** | `start` / `stop` / `restart` / `status` for a local llama-server |
| **Image IR Analyzer** | IMAGE → IMAGE_IR, with low-confidence attributes and debug output |
| **MiniMax H3 Prompt Composer** | An I2VA prompt in the official format |
| **Image IR Prompt Composer** | The generic grounded prompt (CORE / DETAIL / FINAL) |
| **Image IR Grounding Guard** | Rule 9, in `plain` or `minimax_h3` mode |
| **Image IR Write (manual)** | Author IMAGE_IR by hand (advanced / debugging) |
| **Image IR Merge** | Combine two documents; **never overwrites an observation** |
| **Image IR Inspect** | The document, its certainty levels, and the micro-motions it can carry |
| **Image IR Extractor Prompt** | Emit the analysis specification for reading or editing |

Chain the **Grounding Guard** last. A composer audits its own output, but text edited by hand or concatenated with a favourite style string carries no such guarantee.

## The MiniMax H3 format

Reproduced as the official guide specifies it — field names, order, `label: value` punctuation, and the blank line between blocks.

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a woman, seated, ...

overall_soundscape: Rain taps the glass; a chair creaks.

non_diegetic_music: N/A
```

- **Subject motion** and **camera motion** are separate fields. Lens movement is written as natural English inside the shot rather than appended as a label, which is what the guide asks for.
- The two audio fields are **authored**. A still records no sound, so IMAGE_IR cannot ground them and they are not audited for visual content. Empty becomes `N/A`.
- The guard's `minimax_h3` mode preserves field names, markers and the audio blocks while auditing **only** the visual description.

## The analysis prompt

`prompts/image_ir_extractor.txt` — an editable file, not a giant Python constant. It works through subject, face, hair, upper and lower clothing, colours, garment structure, material uncertainty, transparency, hosiery vs bare skin, footwear, hands, pose, leg geometry, environment, objects, background, lighting, framing, camera angle and spatial relationships — and calls out the known risk areas explicitly: subject-left vs viewer-left, fabric hallucination, subtle colour casts, sheer hosiery vs bare legs, background fabric vs wood, crossed-leg geometry, camera angle.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ssain3d-lgtm/ComfyUI-ImageIR-PromptEngine
```

**No extra Python packages.** The engine is standard library only, which is also why it can be imported and tested without ComfyUI:

```python
from imageir import parse_dsl, compose, audit
from imageir.h3 import compose_h3

ir = parse_dsl("subject.identity = a woman\nsubject.gaze = toward camera")
print(compose(ir, motions=["blink"]).final)
print(compose_h3(ir, motions=["blink"]).render())
print(audit(ir, "she turns from looking away toward the camera").report())
```

## Example workflows

- `example_workflows/image-ir-analyze-to-h3.json` — image → analyze → H3 → guard (the default pipeline)
- `example_workflows/image-ir-manual-prompt.json` — hand-written IMAGE_IR → generic prompt → guard
- `example_workflows/image-ir-audit-existing-prompt.json` — audit a prompt you already have

## Not built yet

The schema makes room for these; this pass deliberately does not implement them: a Qwen verifier, automatic confidence routing, automatic multi-image IR merge, face or clothing crops, SAM, pose estimation, model downloading or management.

## Development

```bash
ruff check .
python -m unittest discover -s tests -p "test_*.py" -v
```

External calls are mocked throughout. The tests need no Gemini account, no running LM Studio, no llama.cpp, and no GGUF file.

## License

MIT

## H3 5-mode USER_INTENT pipeline — 한국어 / English

이슈 #5 구현입니다. **현재 이미지 사실은 IMAGE_IR**, **미래 행동·카메라·소리는
USER_INTENT**, **출력 문법은 H3_RULE**로 분리합니다. 기존 노드는 계속 사용할 수 있습니다.

| 새 노드 / New node | 역할 |
|---|---|
| **H3 User Intent Author** | 한국어/영어 요청 → 기존 backend로 구조화·번역 |
| **H3 User Intent from JSON** | USER_INTENT 확인·수정·재입력, 모델 호출 없음 |
| **H3 Reference Pack** | image/video/audio 역할 추가; 여러 노드를 연결 |
| **H3 Mode Router** | 실제 입력에 따라 AUTO 선택, 모호한 입력은 오류 |
| **MiniMax H3 Mode Composer (5 modes)** | 결정적 조립, 출처 trace와 권장 frames 출력 |
| **ImageIR Provenance Guard** | 원본 상태로 재조립하여 미추적 변경 거부/복원 |

### 연결 / Wiring

Backend Config → Analyzer → `first_ir` / `final_ir`

Backend Config + 자유 요청 → H3 User Intent Author → `user_intent`

각 입력을 Router와 Composer에 함께 연결하고 Router의 `mode` 출력을 Composer의
**`routed_mode`** 소켓에 연결합니다. Composer의 `h3_document` → Provenance Guard → 최종 prompt.
처음에는 `shot_count=1`을 권장합니다. 이미지용 기존 Grounding Guard 대신 새
Provenance Guard를 사용해야 요청한 미래 동작이 유지됩니다.

| AUTO 입력 | 모드 | 상태 |
|---|---|---|
| 참조 없음 | T2VA | USER_INTENT 중심, 3개 필드 |
| first_ir | I2VA | 0.00초 이미지 앵커 + 요청한 행동 |
| first_ir + final_ir | FL2VA | 양쪽 이미지 앵커 + 변화 경로 |
| final_ir | L2VA | 마지막 프레임에 도달 |
| reference_pack | Ref2VA | 정확한 6개 섹션, 역할별 참조 |

수동 모드는 intent hint보다 우선합니다. 연결된 자산과 맞지 않는 선택이나 frame IR과
reference pack의 동시 연결은 오류로 처리합니다. 자산을 조용히 버리지 않습니다.
Ref2VA 순서: `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`.

### 스키마 / Schemas

`USER_INTENT_v1`: `raw_request`, `inputs`, `duration_hint`, `mode_hint`,
`shot_count_hint`, `summary`, `requested_actions`, `requested_camera`,
`requested_scene_progression`, `requested_style`, `requested_sound`,
`requested_music`, `constraints`, `dialogue`, `start_states`, `final_states`.
문장 필드는 배열이며 항목은 `{text, input, evidence, shot, path}`입니다.
`evidence`는 지정한 원문 input의 실제 부분 문자열이어야 합니다. 경계 상태는
`path`에 `subject.pose` 같은 IMAGE_IR 경로를 기록합니다. 전체 객체를 JSON으로 저장하고
다시 읽을 수 있습니다. 중간 구조를 확인한 뒤 수정하면 번역 실수를 추적하기 쉽습니다.

`REFERENCE_PACK_v1`: `references` 배열. 항목은 `{label, type, role, subject, image_ir}`.
예: `Picture 1 / image / identity / 1`, `Picture 2 / image / wardrobe / 1`,
`Picture 3 / image / environment / 2`, `Video 1 / video / motion / 1`,
`Audio 1 / audio / sound / 1`. 같은 주체의 여러 참조에는 같은 subject 번호를 사용합니다.
이미지 IR은 optional이며 없으면 시각 세부를 생성하지 않고 경고합니다.
미디어 파일을 첨부하는 노드가 아니므로 실제 H3 생성 workflow에 같은 번호로 자산을 연결해야 합니다.

### 동작 범위 / Guarantees and limits

- IMAGE_IR의 observed/uncertain/absent, confidence, evidence, verification_required와
  viewer-relative geometry를 유지합니다. uncertainty 후보를 확정하거나 negative로 만들지 않습니다.
- 미래 행동은 사용자 출처를 가지므로 이미지의 미세 동작 목록으로 제한하지 않습니다.
  명시적 시작/종료 상태가 이미지와 충돌하면 오류를 냅니다.
- 구조화된 원본으로 재조립하여 변경된 문자열을 검사합니다. 가방·인물 추가 또는
  H3 마커·audio 블록 변경도 추적 없이는 통과하지 않습니다. 텍스트를 바꾸려면 원본 intent를 바꿔 재조립하세요.
- 출처 검증은 일반 자연어의 논리적 함의를 완전히 증명하는 시스템은 아닙니다.
  한국어 번역의 의미, 복잡한 시간 순서 및 다중 주체의 행동은 수동 검토가 필요합니다.
- 24fps, 17k+5 프레임으로 올림합니다. 6초 요청은 158프레임(6.58초).
  362/24초보다 긴 요청은 여러 렌더로 나눠야 합니다. 자동 분할·미디어 편집기는 구현하지 않습니다.
- 샷 수와 시간은 결정적으로 생성합니다. 요청이 짧을 때 새 사건을 추가해 샷을 채우지 않습니다.
  원어 대사는 하나의 `(S1)` 화자로 처리합니다. 여러 화자·언어는 후속 확장 대상입니다.
- 참고 저장소는 실행 시 필요하지 않습니다. 새 패키지 의존성도 없습니다.

### API token

**직접 입력한 api_token은 ComfyUI workflow/PNG metadata에 저장될 수 있습니다.**
기존 파일에 직접 입력했던 키는 해당 파일에서도 제거해야 합니다.
권장: ComfyUI 프로세스 환경에 키를 설정하고 `api_token_env`에 그 **변수 이름만** 입력합니다.
`api_token`은 비워 둡니다. 두 입력이 모두 있거나 지정한 환경변수가 없으면 오류입니다.
새 환경변수는 ComfyUI 재시작 후 반영하세요. 출력/로그의 기존 Secret masking은 유지합니다.

### 참고 분석과 예제

[설계·재사용 분석](docs/h3-integration-design.md)에 검토한 source 커밋과 선택 기준을 기록했습니다.
`example_workflows/image-ir-h3-five-modes.json`은 외부 모델 없이 5개 모드를 실행하는 예제입니다.
실제 한국어 입력은 JSON 노드를 H3 User Intent Author로 바꾸고 Backend Config를 연결합니다.

Windows + ComfyUI 수동 E2E: 노드 로드 → 실제 사진 분석 → 한국어 동작·카메라 authoring →
중간 JSON 확인 → 다섯 모드 생성 → 참조 파일 번호 일치 → H3 렌더의 처음/마지막 프레임 확인.
LM Studio/llama.cpp/Gemini 연결 및 환경변수 키가 workflow에 저장되지 않는지도 확인하세요.
