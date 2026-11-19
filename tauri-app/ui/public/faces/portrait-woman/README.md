# Portrait, woman (`portrait-woman`)

A Heliox face pack (docs/FACE_PACKS.md, format 1) built from still images of one face of nobody real.

Generated with ChatGPT image generation in November 2026; shows no real person. Pack made for Heliox. See README.md in this folder.

## Where the pictures come from

Every picture was generated with ChatGPT image generation (OpenAI) on 2026-11-19: one original portrait from a written
prompt, then edits of that portrait, each asked to change one thing only (the mouth, the eyes, the expression).
The prompts and times, as recorded when they were made:

## Provenance

Generated 2026-11-19, about 16:10 to 16:17, in one ChatGPT chat.
Image model or tool: ChatGPT image generation. The page showed no model name, only the label "Image generated".
No correction prompts were needed.

Template (the text in square brackets and the hair sentence changed per candidate):

"Create a photorealistic studio portrait photograph of a fictional [DESCRIPTION] who does not resemble any real person or celebrity. Head and shoulders, facing the camera straight on, head level and centred, eyes open and looking into the lens, a neutral relaxed expression with the lips gently closed. [HAIR SENTENCE] Soft even light from the front with a faint cool rim light, a plain very dark charcoal background, a plain dark crew-neck top, no jewellery, no glasses, no text, no watermark, no hands. Natural skin texture, sharp focus. Square image, the face filling about half the height of the frame."

Women, hair sentence: "Hair tied back or tucked behind the ears so the whole face, the jawline and the hairline are clear."
Men, hair sentence: "Short neat hair, clean-shaven, so the whole face, the jawline and the hairline are clear."

| File | DESCRIPTION | Size (px) |
|---|---|---|
| woman-1.png | South Asian woman in her late twenties | 1254 x 1254 |
| woman-2.png | East Asian woman in her early thirties | 1254 x 1254 |
| man-1.png | South Asian man in his early thirties | 1254 x 1254 |
| man-2.png | European man in his early thirties | 1254 x 1254 |

Each full prompt is the template with the row's DESCRIPTION and the matching hair sentence filled in, sent word for word.

Each image was generated from the prompt above and shows no real person.

### Variants (2026-11-19, 16:19 to 16:30, same chat)

Edit prompt used in the image viewer's edit box, with the original image selected (CHANGE filled in per variant):

"Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size. The only change: CHANGE."

| File | CHANGE | Size (px) |
|---|---|---|
| woman-1-eyes-closed.png | her eyes are gently closed, as in the middle of a relaxed blink | 1254 x 1254 |
| woman-1-smile.png | a slight, warm, closed-lip smile | 1254 x 1254 |
| woman-1-thinking.png | the eyes look up and a little to one side, as if thinking, with the lips still closed | 1254 x 1254 |

First attempt at the eyes-closed variant, typed in the chat box (not the viewer): "Take the portrait of the South Asian woman that you made earlier in this chat and make one edit to it. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size. The only change: her eyes are gently closed, as in the middle of a relaxed blink." The result changed the face and body position (mean pixel difference from the original about 9.9 against about 2.9 for the kept file), so it was discarded and the viewer-edit version above was kept.

| man-2-eyes-closed.png | his eyes are gently closed, as in the middle of a relaxed blink (applied to man-2.png) | 1254 x 1254 |

No man-1 variants were made. A man-2 smile edit was typed but not submitted.

Each image was generated from the prompt above and shows no real person.

### Variants, second session (2026-11-19, 16:38 to 16:49, a new chat)

Each file was made by attaching the original portrait (woman-1.png or man-2.png) and sending this prompt, with CHANGE filled in as below. The five mouth-shape CHANGE texts are the quoted phrases only; the eyes were already open and looking at the camera in the originals.

"Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size. The only change: CHANGE."

Difference figure = mean absolute pixel difference from the original, out of 255 (all files 1254 x 1254, same size as the original).

| File | Original used | CHANGE | Size (px) | Difference |
|---|---|---|---|---|
| man-2-smile.png | man-2.png | a gentle, clearly visible, closed-lip smile | 1254 x 1254 | 2.45 |
| man-2-thinking.png | man-2.png | the eyes look up and a little to one side, as if thinking, with the lips still closed | 1254 x 1254 | 2.66 |
| woman-1-smile-2.png | woman-1.png | a gentle, clearly visible, closed-lip smile | 1254 x 1254 | 2.85 |
| woman-1-mouth-ah.png | woman-1.png | the mouth is open as if saying 'ah', jaw dropped, upper teeth just visible | 1254 x 1254 | 3.53 |
| woman-1-mouth-ee.png | woman-1.png | the mouth is spread wide as if saying 'ee', teeth together and visible | 1254 x 1254 | 3.49 |

No attempt was discarded. woman-1-smile.png from the first session was left as it was.

Each image was generated from the prompts above and shows no real person.

Each file was made by attaching the original portrait (woman-1.png or man-2.png) and sending this prompt, with CHANGE filled in as below.

"Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size, eyes open and looking at the camera. The only change: CHANGE."

Difference figure = mean absolute pixel difference from the original, out of 255. All files are 1254 x 1254, the same size as the original.

| File | Original used | CHANGE (word for word) | Difference |
|---|---|---|---|
| woman-1-mouth-oh.png | woman-1.png | the lips are rounded as if saying 'oh', mouth half open | 3.52 |
| woman-1-mouth-oo.png | woman-1.png | the lips are pushed forward into a small round opening as if saying 'oo' | 2.82 |
| woman-1-mouth-f.png | woman-1.png | the upper teeth rest on the lower lip as if saying 'f'. The upper front teeth must be clearly visible pressing on the lower lip, with the lower lip tucked slightly under them. (second attempt, extra sentence added) | 3.09 |
| man-2-mouth-ah.png | man-2.png | the mouth is open as if saying 'ah', jaw dropped, upper teeth just visible | 3.15 |
| man-2-mouth-ee.png | man-2.png | the mouth is spread wide as if saying 'ee', teeth together and visible | 2.83 |
| man-2-mouth-oh.png | man-2.png | the lips are rounded as if saying 'oh', mouth half open | 2.62 |
| man-2-mouth-oo.png | man-2.png | the lips are pushed forward into a small round opening as if saying 'oo' | 2.56 |
| man-2-mouth-f.png | man-2.png | the upper teeth rest on the lower lip as if saying 'f' (see note) | 2.68 |
| woman-1-mouth-m.png | woman-1.png | the lips are pressed gently together as if humming 'm' | 3.18 |
| woman-1-mouth-th.png | woman-1.png | the tip of the tongue rests lightly between the upper and lower front teeth as if saying 'th'. The pink tongue tip must be clearly visible between the teeth, with the lips slightly parted. (second attempt, extra sentence added) | 3.17 |

Discarded attempts (kept outside this folder, not delivered):
- woman-1 'f', first attempt, prompt as in the table without the extra sentence: lips slightly parted with teeth showing but not on the lower lip. Difference 2.46.
- woman-1 'th', first attempt, prompt as in the table without the extra sentence: lips slightly parted, no tongue visible. Difference 3.27.

Note on man-2-mouth-f.png: the attachment did not go through for this one (a stale page reference), so the prompt was sent in the same chat without a new attachment and the page edited an earlier portrait from the chat. The result still differs from man-2.png by only 2.68 and looks like the same framing.

Each image was generated from the prompts above and shows no real person.

Third session, continued (same prompt frame as above, one attempt each):

| File | Original used | CHANGE (word for word) | Difference |
|---|---|---|---|
| woman-1-mouth-s.png | woman-1.png | the teeth are almost closed and the lips slightly parted and spread as if saying 's' | 3.01 |
| woman-1-mouth-l.png | woman-1.png | the mouth is slightly open with the tip of the tongue touching just behind the upper front teeth as if saying 'l' | 2.53 |

All files in this session are 1254 x 1254. Not made in this session: woman-1 sh, k, r, eh, uh; all nine man-2 positions (m, th, s, l, sh, k, r, eh, uh); the five soft shapes per face added later. The session was stopped for length, not because of an image limit; no limit or refusal message appeared.

### Fourth session, 2026-11-19 (machine clock, about 17:00 to 17:40 local)

Same prompt frame as above ("Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size, eyes open and looking at the camera. The only change: CHANGE."), original woman-1.png attached each time, one attempt each, one chat.

| File | Original used | CHANGE (word for word) | Size | Difference |
|---|---|---|---|---|
| woman-1-mouth-ah-soft.png | woman-1.png | the mouth is open only a little, as in the middle of relaxed everyday speech, lips apart by about a finger's width, the upper teeth barely visible | 1254 x 1254 | 3.72 |
| woman-1-mouth-ee-soft.png | woman-1.png | the lips are slightly spread and barely parted, as in the middle of relaxed everyday speech on an 'ee' sound, only a hint of the upper teeth, not a smile and not a grin | 1254 x 1254 | 4.10 |

Attempted but not saved: woman-1 'oh' soft. Prompt sent: "the lips are gently rounded and a little open, as in relaxed everyday speech on an 'oh' sound". The image was generated in the chat, but the browser tab stopped answering screenshot requests after the viewer's download click and no file arrived in Downloads, so the session stopped there. No discarded files.

### Fifth session, 2026-11-19 (machine clock, about 17:45 to 18:05 local)

Same prompt frame as before ("Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size, eyes open and looking at the camera. The only change: CHANGE."). For the two half-blink files the words ", eyes open and looking at the camera" were left out. The original portrait was attached each time, one attempt each. Files 1 to 10 came from one chat; the first tab stopped answering screenshots while file 11 was generating, so a second tab with a new chat made files 11 and 12.

| File | Original used | CHANGE (word for word) | Size | Difference |
|---|---|---|---|---|
| woman-1-mouth-oh-soft.png | woman-1.png | the lips are gently rounded and a little open, as in relaxed everyday speech on an 'oh' sound | 1254 x 1254 | 3.32 |
| woman-1-mouth-oo-soft.png | woman-1.png | the lips are gently rounded and nearly closed, as in relaxed everyday speech on an 'oo' sound, not a pucker and not a kiss | 1254 x 1254 | 2.39 |
| woman-1-mouth-eh.png | woman-1.png | the mouth is a little open and relaxed, as in relaxed everyday speech on the 'e' of the word 'bed', teeth slightly apart | 1254 x 1254 | 2.43 |
| man-2-mouth-ah-soft.png | man-2.png | the mouth is open only a little, as in the middle of relaxed everyday speech, lips apart by about a finger's width, the upper teeth barely visible | 1254 x 1254 | 2.70 |
| man-2-mouth-ee-soft.png | man-2.png | the lips are slightly spread and barely parted, as in the middle of relaxed everyday speech on an 'ee' sound, only a hint of the upper teeth, not a smile and not a grin | 1254 x 1254 | 2.09 |
| man-2-mouth-oh-soft.png | man-2.png | the lips are gently rounded and a little open, as in relaxed everyday speech on an 'oh' sound | 1254 x 1254 | 2.29 |
| man-2-mouth-oo-soft.png | man-2.png | the lips are gently rounded and nearly closed, as in relaxed everyday speech on an 'oo' sound, not a pucker and not a kiss | 1254 x 1254 | 2.36 |
| man-2-mouth-eh.png | man-2.png | the mouth is a little open and relaxed, as in relaxed everyday speech on the 'e' of the word 'bed', teeth slightly apart | 1254 x 1254 | 2.77 |
| woman-1-eyes-half.png | woman-1.png | the eyelids are half lowered, as in the middle of a blink, with the eyes still partly visible and the lips closed (prompt without the eyes-open words) | 1254 x 1254 | 3.08 |
| man-2-eyes-half.png | man-2.png | the eyelids are half lowered, as in the middle of a blink, with the eyes still partly visible and the lips closed (prompt without the eyes-open words) | 1254 x 1254 | 2.32 |
| man-2-mouth-m.png | man-2.png | the lips are pressed gently together as if humming 'm' | 1254 x 1254 | 2.16 |
| man-2-mouth-s.png | man-2.png | the teeth are almost closed and the lips slightly parted and spread as if saying 's' | 1254 x 1254 | 2.38 |

Discarded attempts: none. Note: the first 'm' attempt (first tab) generated an image, but the tab froze before the download, so the 'm' prompt was sent again in the second tab; the saved file is from the second chat.

### Sixth session, 2026-11-19 (machine clock, about 18:35 to 18:50 local)

Same prompt frame as before ("Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size, eyes open and looking at the camera. The only change: CHANGE."). The original portrait was attached each time, one attempt each. The first tab stopped answering screenshots after the second message was sent in its chat, so a second tab with a new chat made the 'l' file; the second tab also stopped answering after the third message, so the session stopped there.

| File | Original used | CHANGE (word for word) | Size | Difference |
|---|---|---|---|---|
| woman-1-mouth-f-soft.png | woman-1.png | the lower lip lightly touches the upper teeth as if starting to say 'f', the teeth only just visible, the jaw relaxed | 1254 x 1254 | 2.86 |
| woman-1-mouth-l-soft.png | woman-1.png | the lips are slightly parted as if saying 'l' in ordinary speech, the tongue not visible | 1254 x 1254 | 2.65 |

Not saved: woman-1-mouth-m-soft.png. Prompt sent (second tab): "the lips are gently closed and relaxed, as if humming 'm', not pressed and not pulled down". The second tab stopped answering screenshot requests while it generated, so no image was downloaded.

Discarded attempts: none.

## Files used for this pack

| Source file | Used as | SHA-256 |
|---|---|---|
| `woman-1.png` | sil (closed mouth), idle frame | `e9765fb76ae87d59d756c529a49e8530621715084b2ec13bac884aeb9412ba5a` |
| `woman-1-mouth-f-soft.png` | mouth FF | `90579ddf7884c7f930d138b6b466a699de3b6b4acd3fee96d9183039a8a935ec` |
| `woman-1-mouth-s.png` | mouth SS | `324b6c399279349fdca0e46d283212237e90f3df35897a817251702f4790e808` |
| `woman-1-mouth-l-soft.png` | mouth nn | `531197689c5fdac55e9fe9b41c3e1da7bf73eb66c7d15b5b8d45e7b66b190e84` |
| `woman-1-mouth-ah-soft.png` | mouth aa | `15ab82d2961ac5388e92d1ddf5abc2c3dab5787080d7f249fa700b882f5bc27a` |
| `woman-1-mouth-eh.png` | mouth E | `5126108348f83442b4539ad8511bbc0ef5655d2bd08a512f36892572a11d9fb7` |
| `woman-1-mouth-ee-soft.png` | mouth I | `e7510dfff0a5a28f7661b6be42a21c70106fd7d6ffbf8648eaa15cddcb28405f` |
| `woman-1-mouth-oh-soft.png` | mouth O | `be3cc50c4817498448e202eb7245daa12c15d43a890671bfaf7dedfc5c3e3eb5` |
| `woman-1-mouth-oo-soft.png` | mouth U | `14a567f0cb4fd22654cbad3e7f5f61d3faf7ae1d949f280e4cab525041eb6815` |
| `woman-1-eyes-closed.png` | blink (closed) | `b341eebef28c9c7d710bde8726438ea7cff6d81469b9cef3ddbeb0c01d1ad9b2` |
| `woman-1-eyes-half.png` | blink (half) | `b6addf6bbee801143ef5a09a54e0da885fc72e129e72c3f599061c6229b88bfe` |
| `woman-1-thinking.png` | eyes.thinking | `71e27ea79a360f84f66f1dc096f8cd914925f3eef6ca7302d61b60a6b4640b74` |
| `woman-1-smile-2.png` | not used (no place in format 1) | `c7705cd491c6a704911311c22de76047e1b0cc34c7d824776f5cd84a836696a3` |

## Terms

This pack is distributed with Heliox OS under the project's own licence, the MIT License (the `LICENSE` file at the
root of the repository). The pictures are generated and show no real person.

OpenAI's Terms of Use (https://openai.com/policies/row-terms-of-use/, effective 1 January 2026, read on 2026-11-19),
section "Content", say that as between the user and OpenAI, "you (a) retain your ownership rights in Input and (b) own
the Output", and that OpenAI assigns to the user whatever rights it has in the Output. The same terms forbid
presenting Output as made by a human when it was not; this README and the `credit` line say plainly that the pictures
are generated. In several countries a picture made by an AI without human authorship may carry no copyright at all.

## How the pack was made

`face_lab/scripts/build_pack_from_stills.py` (CPU image work only, no neural model except OpenCV's 230 KB YuNet face
detector used to place the rectangles): each edit registered onto the original by an affine ECC fit on the upper
face; a 512 x 512 square around the head; the idle frame saved as WebP q80; every mouth and eye image
tone-matched to the decoded idle and kept identical to it wherever the edit did not change the face; one shared
feathered alpha per layer. `mouth-sil.webp` is cut from the decoded idle itself (lossless), so a pause shows exactly
the idle frame. Build report: `face_lab/packs/_woman-1_work/build_report.json`.

## Choices in this build

- Left out on purpose (the player shows its stand-in): TH, PP.
- `SS` toned down: the morph 80% of the way from the closed mouth is the full shape.
- `mouth_steps`: in-between images made by morphing (dense optical flow both ways, both images warped toward each other and blended; the inside of the mouth is carried from the open still rather than faded in), at one third and two thirds, or one half for shapes whose lips move less than 3 px. `FF` has none: every morph of teeth onto the lower lip looked pasted at 3x.
