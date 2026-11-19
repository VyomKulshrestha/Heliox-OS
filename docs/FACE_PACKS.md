# Face packs

Heliox can show a face beside the conversation that moves its mouth while Heliox speaks (Settings, Appearance, Face;
off by default). The face is a *pack*: a folder of images and a `manifest.json`. This page is everything needed to
make one. No code changes are needed to add a pack.

## How the face works

The window draws the pack on a 2D canvas: an idle picture (one frame or a short loop), the eyes on top of it, and a
mouth on top of that. For every piece of speech the daemon sends a timeline of mouth shapes taken from the voice's own
phoneme timing, and the window shows the right mouth image at each moment, blending into the next one over about
70 ms. When the voice gives no phoneme timing, the mouth opens and closes with the loudness of the audio instead
(closed, `E`, `aa`). With the system voice there is no audio to time a mouth by, so the face stays still.

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
`packs.json` and the pack appears in Settings under the name its manifest gives. Any face can be used with any voice.

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
  "states": { "listening": { "tilt": -2 }, "thinking": { "tilt": 1.5, "lift": -2 } },
  "breathe": { "amount": 0.004, "seconds": 4.5 }
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
| `eyes.blink` | no | The blink, as eye images played at `blink_fps`, every `blink_every` seconds (a random time between the two). |
| `eyes.listening`, `eyes.thinking` | no | Eyes for those moments (raised brows, a look up). |
| `eyes.rect` | no | Where eye images go: `[x, y, width, height]`. Default: the whole picture. |
| `mouth.shapes` | yes | One image per mouth shape (below). `sil`, the closed mouth, is required. |
| `mouth.rect` | no | Where mouth images go: `[x, y, width, height]`. Default: the whole picture. |
| `states` | no | A small turn of the whole head for `listening`, `thinking` and `speaking`: `tilt` in degrees, `lift` in pixels (up is negative), `scale`. |
| `breathe` | no | A slow scale of the whole picture: `amount` (0.004 is 0.4 %) over `seconds`. Leave it out for a filmed loop. |
| `tint` | no | For drawings only: `{"idle": "--text-secondary", "eyes": "--text-primary", "mouth": "--text-primary"}`. Each layer's image is used as a mask and filled with that colour from the window's theme, so a line drawing follows light and dark mode. Leave it out for a photo. |

Image files may be WebP, PNG, JPEG, AVIF or SVG. File names: letters, digits, dots, dashes and underscores.

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
- **Blinks:** 2 or 3 images of the eye region (half closed, closed), cut the same way, with feathered edges.
- **Size on disk:** WebP at quality 80 keeps a pack of this kind to a few megabytes.

## The placeholder

`public/faces/placeholder` is a line drawing in SVG (curves and ellipses written for Heliox, no source image),
released under CC0 1.0. It is plainly a
drawing and draws in the window's own colours (`tint`). It exists to prove the timing until a realistic pack ships.

## The portraits

`public/faces/portrait-woman` ("Portrait, woman") and `public/faces/portrait-man` ("Portrait, man") are photographic
packs made from generated portraits of people who do not exist: one still idle frame, a blink, thinking eyes and the
mouth shapes their makers had stills for (the others are drawn with stand-ins). Each folder's `README.md` records how
every picture was made and states the terms: the pack is distributed with Heliox OS under the project's MIT License.
A pack may be rebuilt in place (same folder, same id) with more pictures; nothing in the code names its files.
