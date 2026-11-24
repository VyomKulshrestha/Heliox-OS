# Face packs

Heliox can show a face beside the conversation that moves its mouth while Heliox speaks (Settings, Appearance, Face;
off by default). The face is a *pack*: a folder of images and a `manifest.json`. This page is everything needed to
make one. No code changes are needed to add a pack; one that names a voice also adds a line to the daemon's table
of pack voices ("The face's voice").

## How the face works

The window draws the pack on a 2D canvas: an idle picture (one frame or a short loop), the eyes on top of it, and a
mouth on top of that. For every piece of speech the daemon sends a timeline of mouth shapes taken from the voice's own
phoneme timing, with the loudness of the audio (25 values a second). When the voice gives no phoneme timing, the
loudness alone moves the mouth (half open `E`, open `aa`). With the system voice there is no audio to time a mouth by,
so the face stays still.

The mouth is a weight for each shape, from 0 (at rest) to 1 (fully formed), and the window moves the weights the way
a mouth moves in running speech:

- each moment of the timeline asks for one shape, as strongly as the sound is loud: a quiet syllable opens the mouth
  less (about half at the quietest), and the mouth relaxes as the sound dies;
- every weight moves toward what is asked like a lightly damped spring with a time constant of about 30 ms (a
  little faster when a shape gives way), so a short sound pulls the mouth only part of the way and never reaches a
  wide shape, and one shape gives way to the next without a jump;
- a lip closure (p, b, m) moves about three times faster and always closes fully, on time: it is what the eye checks
  sync against; the lips stay pressed (`PP`) for at most 140 ms, then rest closed, so a held m does not look pursed;
- the targets run about 50 ms ahead of the sound, the spring's own delay plus half a frame, so the picture lands on
  the sound;
- wherever the sound is quiet (3 or less on the 0 to 15 loudness scale) for 80 ms or more, such as a pause or a
  drawn-out last vowel that has faded, the mouth closes as the sound stops and opens as it returns; a shorter quiet
  (a stop inside a word) changes nothing;
- a shape the pack has no picture of stays calm: a lip closure (`PP`) shows the closed mouth; t, d and th use the
  pack's `nn` (a small calm opening) and ch its `SS` when it has them; other consonants keep the opening of the vowel
  before them at 40 % (or a small neutral opening); a vowel uses the nearest vowel the pack has. A consonant with a
  picture of its own is formed at most 45 % of the way, so the vowels carry the opening: at 70 % a consonant picture
  that opens nearly as far as a vowel (the woman's `nn`) left her stressed vowels hardly more open than her n and t;
- a cut closes the mouth at once: the face draws a frame at rest as the cut notice arrives, without waiting for its
  next frame, and keeps the mouth at rest for whatever the reply would still have said;
- one picture is always dominant: between two steps of a shape the blend stays on one step until a quarter of the
  way and is on the next from three quarters (a lip closure's steps cross between 40 and 60 %), and of two shapes
  drawn at once the less formed one fades in only as the two near each other, half and half only where they swap.
  Both are drawn straight onto the face, the more formed one exactly as when it is alone, so a closing mouth looks
  closed as soon as its shapes are. Two pictures with the chin at different heights blended half and half show two
  soft chins; with `mouth.chin` the less formed shape is also drawn at the step whose chin matches the other's, as
  far as it reaches.

The rest of the face moves with the speech: the strongest syllables, at most one every 600 ms, bring a nod of
about 2.5 px at the 132 px the face is shown at (1.9 % of the picture's height), rising and falling over about
600 ms with its lowest point 80 ms before the syllable, and a lean of up to 0.8 degrees that changes side every second
nod. That is slow enough (a dozen pixels a second at most) to read as a person talking rather than as movement; a
nod for every loud syllable, as before, bobs once it is large enough to be seen. A blink never
starts while a word sounds, and the end of a phrase brings one more often; a long pause may bring a short glance
through the pack's `eyes.glance` pictures and back; and at rest the face breathes (`breathe`). Under reduced motion the head and the eyes
stay still and the mouth only opens and closes.

Nothing is generated while Heliox runs: every picture comes from the pack. The face needs no graphics card.

## Where a pack lives

```
tauri-app/ui/public/faces/
  packs.json                 { "packs": ["placeholder", "your-pack"] }
  your-pack/
    manifest.json
    idle-00.webp ...
    mouth-sil.webp ...
```

The folder name is the pack's id: lower-case letters, digits and dashes, at most 40 characters. Add the id to
`packs.json` and the pack appears in Settings under the name its manifest gives. Any face can be used with any voice;
a pack may name the voice that suits it (`voice`, below, and "The face's voice" for the one line the daemon needs).

## manifest.json

```json
{
  "format": 1,
  "id": "your-pack",
  "name": "Ana",
  "credit": "Made for Heliox by <who>, from <how>. <licence>.",
  "kind": "photo",
  "size": [512, 512],
  "idle": { "frames": ["idle-00.webp", "idle-01.webp"], "fps": 12, "loop": "pingpong",
            "mouth_offsets": [[0, 0], [1, -1]] },
  "eyes": { "blink": ["blink-1.webp", "blink-2.webp", "blink-1.webp"], "blink_fps": 24, "blink_every": [2.5, 6],
            "rect": [150, 170, 212, 80] },
  "mouth": { "rect": [176, 300, 160, 112], "shapes": { "sil": "mouth-sil.webp", "aa": "mouth-aa.webp" } },
  "states": { "listening": { "tilt": -2, "mouth": "mouth-smile.webp" }, "thinking": { "tilt": 1.5, "lift": -2 } },
  "breathe": { "amount": 0.004, "seconds": 4.5 },
  "voice": { "kokoro": "af_heart", "hint": "feminine" }
}
```

| Field | Required | Meaning |
|---|---|---|
| `format` | yes | Always `1`. |
| `id` | yes | The folder's name. |
| `name` | yes | What Settings shows, 1 to 40 characters. |
| `credit` | yes | Who made it, how, and its licence, in one line (up to 200 characters). Shown in Settings. |
| `kind` | no | `"drawing"` or `"photo"`; information only. |
| `size` | yes | Width and height of the picture in pixels. Every rectangle below is in these pixels. |
| `idle.frames` | yes | One image, or a loop. A loop plays at `fps`, forward and back (`"pingpong"`, the default) or forward only. |
| `idle.mouth_offsets` | no | One `[dx, dy]` per idle frame: how far the mouth moved in that frame, so the mouth images follow the head. |
| `eyes.open` | no | An eyes layer drawn over the idle frame (for drawn faces; a photo has its eyes in the idle frames). |
| `eyes.blink` | no | The blink, as eye images played at `blink_fps`, every `blink_every` seconds (a random time between the two). A blink of one image (closed eyes only) lasts `1 / blink_fps` seconds and fades in and out over the open eyes instead of snapping shut; add a half-closed image (`["blink-half.webp", "blink-closed.webp", "blink-half.webp"]`) and the images play in turn. |
| `eyes.glance` | no | A small shift of the eyes, as pictures from slight to further (`["eyes-glance-1.webp", "eyes-glance-2.webp"]`), cut like the blink. Shown now and then in a pause between phrases while Heliox speaks: out through the list, a moment at the last, and back. |
| `eyes.listening`, `eyes.thinking` | no | Eyes for those moments (raised brows, a look up). |
| `eyes.rect` | no | Where eye images go: `[x, y, width, height]`. Default: the whole picture. |
| `mouth.shapes` | yes | One image per mouth shape (below). `sil`, the closed mouth, is required. |
| `mouth.rect` | no | Where mouth images go: `[x, y, width, height]`. Default: the whole picture. A rectangle covering the whole lower face (the jaw moving with each shape) works as well as one around the lips; the canvas draws at most about ten pictures of it a frame. |
| `mouth.chin` | no | How far each shape's full picture moves the chin, in the pack's pixels, down positive: `{"aa": 10.7, "O": 4.9, "PP": -2.8}`. When two shapes are drawn at once (one giving way to the next), the less formed one is drawn at the weight whose chin sits where the other's does, so the face shows one chin, not two half-faded ones. Shapes left out, or moving the chin less than a pixel or the other way, are drawn as formed. |
| `states` | no | How the face rests in `idle`, `listening`, `thinking` and `speaking`: `tilt` in degrees, `lift` in pixels (up is negative), `scale` (0.5 to 2) for a small turn of the whole head, and `mouth`, an image drawn in `mouth.rect` in place of `sil` while the face rests in that state (a closed-lip smile while listening, for example; missing means `sil`). Every one of them eases in over about 160 ms when the state changes. While Heliox speaks, the closed moments of speech show the `speaking` state's `mouth` (normally left out, so `sil`). |
| `mouth_steps` | no | In-between pictures for a shape: `{"aa": ["mouth-aa-1.webp", "mouth-aa-2.webp"]}`, ordered from nearly closed to nearly full, made with the same crop and alpha as the shape's own picture (morphed offline). The shape's own picture is the last step and the resting mouth is step zero. A shape formed part of the way shows the two neighbouring steps at that point blended, so the mouth opens through real in-between shapes instead of fading a full one in. A shape without steps fades its full picture in over the rest. Each listed shape must have its own picture in `mouth.shapes`; 1 to 8 steps. |
| `overscan` | no | How much larger than its card the picture is drawn, from 0 to 0.25 (0.04 is 4 %), about its centre. Left out, a `"photo"` pack gets exactly enough that no `tilt`, `lift` or `scale` in `states`, with breathing, ever shows the card behind the picture (about 2 % for a 3 px lift, about 10 % for a 3 degree tilt), and a drawing gets none. Give `0` to turn it off, or a number to fix it. |
| `voice` | no | The voice that suits the face: `kokoro`, a Kokoro voice id (`af_heart`, `am_michael`, `bm_lewis`; the first letter is the language, the second `f` or `m`), and `hint`, `"feminine"` or `"masculine"`, for when that exact voice cannot speak on the computer. See "The face's voice" below. |
| `breathe` | no | A slow scale of the whole picture: `amount` (0.004 is 0.4 %) over `seconds`. Leave it out for a filmed loop. |
| `tint` | no | For drawings only: `{"idle": "--text-secondary", "eyes": "--text-primary", "mouth": "--text-primary"}`. Each layer's image is used as a mask and filled with that colour from the window's theme, so a line drawing follows light and dark mode. Leave it out for a photo. |

Image files may be WebP, PNG, JPEG, AVIF or SVG. File names: letters, digits, dots, dashes and underscores.

## The face's voice

When a person chooses a face in Settings whose pack names a voice:

- if they have never chosen a voice themselves, Heliox switches to the face's voice with the face, and the Face row
  says so in one sentence. Heliox knows a voice was never chosen from `voice.tts_voice_chosen`, which becomes true the
  first time the person picks a voice (Settings, Voice; a spoken setting; the voice for their language; the Face
  row's offer). A config file saved before that field existed counts as chosen, since every save wrote the voice and
  the default may have been picked on purpose;
- if they have chosen one, it is kept, and the row offers the face's voice with one button;
- with Pocket TTS, the face uses the Pocket voice approved for its kind (`POCKET_BY_KIND` in
  `pilot/system/face_voice.py`, one column per engine in `PACK_VOICES`): alba for a feminine face. No masculine Pocket
  voice is approved yet, so a masculine face brings its Kokoro voice instead of keeping a voice of the wrong kind:
  switched with the face when Kokoro and the voice are on disk and the person never chose a voice, otherwise offered
  with what it downloads (about 330 MB the first time, when Kokoro is not on the computer). Until the person accepts
  that download, the voice that was speaking stays;
- if the face's voice cannot speak here, nothing is switched: with the system voice the row says to
  choose Kokoro TTS under Voice; when the voice file is not on disk yet the row offers it, and Kokoro fetches it
  (about half a megabyte) the first time it speaks, the way the voice for another language is added; when Heliox
  speaks another language, a Kokoro voice of that language matching `hint` is offered if there is one.

Choosing "Off" changes no voice, and a voice set by hand afterwards stays.

The daemon applies this rule itself whenever `face.pack` is set, whatever set it: the Face row, another window or
client through the settings update, a spoken setting, or `config.toml` edited by hand (read when Heliox starts, or
when the edited file is picked up while it runs). The Face row only shows what happened and offers the voice. The
daemon cannot read the window's packs, so it keeps a copy of each pack's `voice` in `PACK_VOICES`
(`daemon/pilot/system/face_voice.py`); a pack that names a voice is added there too, and a test fails until the two
agree.

## The fifteen mouth shapes

| Shape | Sounds | Mouth |
|---|---|---|
| `sil` | silence | closed, relaxed |
| `PP` | p, b, m | lips pressed together |
| `FF` | f, v | lower lip under the upper teeth |
| `TH` | th | tongue tip between the teeth |
| `DD` | t, d, l | slightly open, tongue behind the teeth |
| `kk` | k, g, h, ng | open about a third, tongue back |
| `CH` | sh, ch, j, zh | lips pushed forward, teeth close |
| `SS` | s, z | teeth together, lips apart |
| `nn` | n | slightly open |
| `RR` | r | lips slightly rounded |
| `aa` | a in "father", "cat" | wide open |
| `E` | e in "bed", the "uh" sound | half open |
| `I` | ee in "see", y | spread, a little open |
| `O` | o in "go", "law" | round, open |
| `U` | oo in "food", w | small round |

A pack may leave shapes out; each missing one is drawn with the nearest one it has (for example `TH` with `DD`, `CH`
with `SS`, `U` with `O`), then `E`, then `sil`. A pack with `sil`, `PP`, `E`, `aa`, `O` and `U` looks reasonable; all
fifteen look best.

## Making a realistic pack

A realistic face is one newly made face of nobody real, with its rights written down in `credit` and in the pack's
own `LICENSE` or `README` next to the manifest: the tool that made it, the date, the prompt or source, and the terms
that let it be shipped in an open-source app. A face of a real person, or one whose maker's terms are unknown, cannot
be used.

- **Size:** 512 x 512 for the idle frames is plenty (the window shows the face at 104 to 132 CSS pixels, twice that on
  a high-density screen). Mouth images can be much smaller: a crop of about 160 x 112 around the mouth and jaw.
- **The idle loop:** 2 to 4 seconds at 12 frames a second (24 to 48 frames), mouth closed and relaxed in every frame,
  looking at the camera. Keep the head nearly still: small movements of a few pixels at most, and either no movement
  of the jaw at all or `mouth_offsets` for every frame. A loop played forward and back hides the seam.
- **The mouth images:** render each of the fifteen shapes on the same face, lit and coloured like the idle loop, then
  cut the same rectangle around the mouth from each. Give each image soft (feathered) transparent edges a few pixels
  wide, so it blends into the face under it. The closed mouth `sil` must match the idle frames, or the face will flicker
  at every pause.
- **In-between steps:** for each shape, one to four pictures between the closed mouth and the full shape
  (`mouth_steps`), made by morphing, same crop and alpha. With a moving jaw, four for the wide shapes keep the chin
  sharp half way between two of them; one or two for small shapes.
- **Calm consonants:** keep consonant pictures (`DD`, `nn`, `TH`, `SS`, `FF`) close to a relaxed half-open mouth. A
  showy one (a visible tongue, prominent teeth) flashes on the commonest sounds; leaving a consonant out is better
  than a showy picture, since the window keeps the vowel's opening instead.
- **Blinks:** 2 or 3 images of the eye region (half closed, closed), cut the same way, with feathered edges. One
  closed image works too (it fades in and out).
- **A resting smile:** a closed-lip smile cut like the mouth images, named in `states.listening.mouth` (or `idle`).
  It must match the idle frame everywhere but the mouth, like `sil`.
- **Tilt:** a photograph is enlarged to hide its corners while it tilts, which crops a little of its edges: keep
  `tilt` to 2 degrees or less, or use `lift` alone.
- **Size on disk:** WebP at quality 80 keeps a pack of this kind to a few megabytes.

## The placeholder

`public/faces/placeholder` is a line drawing in SVG (curves and ellipses written for Heliox, no source image),
released under CC0 1.0. It is plainly a
drawing and draws in the window's own colours (`tint`). It exists to prove the timing until a realistic pack ships.

## The portraits

`public/faces/portrait-woman` ("Portrait, woman") and `public/faces/portrait-man` ("Portrait, man") are photographic
packs made from generated portraits of people who do not exist: one still idle frame, a blink, thinking eyes and the
mouth shapes their makers had stills for (the others are drawn with stand-ins). The woman names the voice `af_heart`,
the man `am_michael`. Each folder's `README.md` records how
every picture was made and states the terms: the pack is distributed with Heliox OS under the project's MIT License.
A pack may be rebuilt in place (same folder, same id) with more pictures; nothing in the code names its files.
