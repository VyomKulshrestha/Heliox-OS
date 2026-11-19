# Portrait, woman (`portrait-woman`)

A Heliox face pack (docs/FACE_PACKS.md, format 1): a photographic portrait of a person who does not exist, with
mouth pictures for speech, blink pictures and thinking eyes.

Generated with ChatGPT image generation in November 2026. Shows no real person.

## Where the pictures come from

Every source picture was generated with ChatGPT image generation (OpenAI) on 19 November 2026. The page showed no
model name. First an original portrait was made from this prompt, with DESCRIPTION "South Asian woman in her late twenties" and
the hair sentence "Hair tied back or tucked behind the ears so the whole face, the jawline and the hairline are clear.":

> Create a photorealistic studio portrait photograph of a fictional [DESCRIPTION] who does not resemble any real person or celebrity. Head and shoulders, facing the camera straight on, head level and centred, eyes open and looking into the lens, a neutral relaxed expression with the lips gently closed. [HAIR SENTENCE] Soft even light from the front with a faint cool rim light, a plain very dark charcoal background, a plain dark crew-neck top, no jewellery, no glasses, no text, no watermark, no hands. Natural skin texture, sharp focus. Square image, the face filling about half the height of the frame.

Every other picture is an edit of that original, made by attaching it and sending this prompt with CHANGE filled in
(for the blink pictures without the words "eyes open and looking at the camera"; a few early edits used the same
prompt without them as well):

> Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size, eyes open and looking at the camera. The only change: CHANGE.

No picture was retouched by hand.

## Source pictures used

| File | Used for | CHANGE | SHA-256 |
|---|---|---|---|
| `woman-1.png` | the idle picture and the closed mouth (`sil`) | the original portrait | `e9765fb76ae87d59d756c529a49e8530621715084b2ec13bac884aeb9412ba5a` |
| `woman-1-mouth-f-soft.png` | mouth `FF` | the lower lip lightly touches the upper teeth as if starting to say 'f', the teeth only just visible, the jaw relaxed | `90579ddf7884c7f930d138b6b466a699de3b6b4acd3fee96d9183039a8a935ec` |
| `woman-1-mouth-s.png` | mouth `SS` | the teeth are almost closed and the lips slightly parted and spread as if saying 's' | `324b6c399279349fdca0e46d283212237e90f3df35897a817251702f4790e808` |
| `woman-1-mouth-l-soft.png` | mouth `nn` | the lips are slightly parted as if saying 'l' in ordinary speech, the tongue not visible | `531197689c5fdac55e9fe9b41c3e1da7bf73eb66c7d15b5b8d45e7b66b190e84` |
| `woman-1-mouth-ah-soft.png` | mouth `aa` | the mouth is open only a little, as in the middle of relaxed everyday speech, lips apart by about a finger's width, the upper teeth barely visible | `15ab82d2961ac5388e92d1ddf5abc2c3dab5787080d7f249fa700b882f5bc27a` |
| `woman-1-mouth-eh.png` | mouth `E` | the mouth is a little open and relaxed, as in relaxed everyday speech on the 'e' of the word 'bed', teeth slightly apart | `5126108348f83442b4539ad8511bbc0ef5655d2bd08a512f36892572a11d9fb7` |
| `woman-1-mouth-ee-soft.png` | mouth `I` | the lips are slightly spread and barely parted, as in the middle of relaxed everyday speech on an 'ee' sound, only a hint of the upper teeth, not a smile and not a grin | `e7510dfff0a5a28f7661b6be42a21c70106fd7d6ffbf8648eaa15cddcb28405f` |
| `woman-1-mouth-oh-soft.png` | mouth `O` | the lips are gently rounded and a little open, as in relaxed everyday speech on an 'oh' sound | `be3cc50c4817498448e202eb7245daa12c15d43a890671bfaf7dedfc5c3e3eb5` |
| `woman-1-mouth-oo-soft.png` | mouth `U` | the lips are gently rounded and nearly closed, as in relaxed everyday speech on an 'oo' sound, not a pucker and not a kiss | `14a567f0cb4fd22654cbad3e7f5f61d3faf7ae1d949f280e4cab525041eb6815` |
| `woman-1-eyes-closed.png` | blink (closed) | her eyes are gently closed, as in the middle of a relaxed blink | `b341eebef28c9c7d710bde8726438ea7cff6d81469b9cef3ddbeb0c01d1ad9b2` |
| `woman-1-eyes-half.png` | blink (half) | the eyelids are half lowered, as in the middle of a blink, with the eyes still partly visible and the lips closed | `b6addf6bbee801143ef5a09a54e0da885fc72e129e72c3f599061c6229b88bfe` |
| `woman-1-thinking.png` | thinking eyes and the glance | the eyes look up and a little to one side, as if thinking, with the lips still closed | `71e27ea79a360f84f66f1dc096f8cd914925f3eef6ca7302d61b60a6b4640b74` |

## How the pack was made

Processor-only image work with OpenCV (no neural network beyond OpenCV's small YuNet face detector, used only to
place the rectangles). Each edited picture was aligned to the original on the forehead, nose and cheeks, and the
head was cut to a 512 x 512 square (`idle-00.webp`, WebP quality 80).

Each mouth picture covers the whole lower face, `mouth.rect` [126, 285, 276, 186], from the nose to below the chin,
with a feathered edge. Its skin is the idle picture's own pixels, moved: the motion between the original and that
shape's edited picture was measured with dense optical flow, cleaned (smoothed, the leftover misalignment removed,
movement smaller than the edits' own drift dropped) and faded to nothing well inside the feathered edge, and the idle
picture was warped by it, so the jaw, chin, cheeks and the folds beside the mouth move as in the edit while the skin
keeps the idle picture's texture and tone. Only the lips and the inside of the mouth (teeth, tongue, the gap) come
from the edited picture. The in-between pictures (`mouth_steps`) take the same motion part of the way, with lips
morphed between the closed and the open mouth: a fifth, two, three and four fifths where the chin or the lower lip
moves 6 px or more, a third and two thirds where it moves 3 px or more, and half of it below that.
`mouth-sil.webp` is the idle picture itself, so a closed mouth is identical to the idle frame. The eye pictures are
the edits' eye region on the idle picture; the two glance pictures are the thinking eyes a sixth and a third of the way.

- `mouth.chin` gives each shape's chin travel at the full shape in pack pixels, down positive.
- `SS` is drawn 80% of the way from the closed mouth to its still, lips and jaw alike.
- `FF` has no in-between pictures: teeth meeting the lower lip did not morph cleanly.
- Shapes without a picture of their own (the app shows the closed mouth or the nearest shape the pack has): `PP`, `TH`, `DD`, `kk`, `CH`, `RR`.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `blink-closed.webp` | 5,858 | `83c3fba5edb7eb49ece4592b47f2a4a07bc8137de131c1716f75f31bc55cab64` |
| `blink-half.webp` | 5,878 | `998ec2ed75a26dc53a3d488c57f6651e7f733b1c0975c371da5f430af0d3fa6e` |
| `eyes-glance-1.webp` | 6,048 | `1eaba82ca71698955e99a4d999c476ad16ec8c703cd16f059cbf49aeb62bfa9e` |
| `eyes-glance-2.webp` | 6,048 | `2ed5baa400924e73e4718a99b8e568f9629c3b9b9b0fa2b15f733e4c6b3c8cdd` |
| `eyes-thinking.webp` | 6,184 | `e1920900b6709b1471bf122e33d51d5f1fdff1d67390e66785536833f7219a60` |
| `idle-00.webp` | 13,332 | `9b6fc4e2214b6b0140ff9d62e3e461b6189916f5ec8cf01c5e47e7b4a2fe2e1b` |
| `manifest.json` | 2,003 | `c0ad3e65a136167ec1021977f814c9c0528c2de7c62d94f35e22fd8f1abefc09` |
| `mouth-aa-1.webp` | 41,958 | `2f8fec4b4e9579e0e009cb087f4697d57bf25e0b432659930d99b741616f82f2` |
| `mouth-aa-2.webp` | 42,140 | `1e332ccbe021f77312ea22cc253d6a879f5d2b566c67a3735ee3e8890d5082d2` |
| `mouth-aa-3.webp` | 42,460 | `41dbc7100a44bc3ef1c873f5574e87804f60e2e6db11085c1af07e77354e2d04` |
| `mouth-aa-4.webp` | 42,496 | `3eb97f5a56ba231f3f52b4c3ceab8dffc744c08857a7cb870ed041e1c008be4a` |
| `mouth-aa.webp` | 42,752 | `52d0aa1e6c1ecdc081ac3f4c917de9c8b9e532f8ed2d5447642198fd584e25b8` |
| `mouth-E-1.webp` | 41,916 | `52d576d2d3393398e6ca7443d7e76ab67fbf52e6df6e6a831a137e819554a8b2` |
| `mouth-E-2.webp` | 42,036 | `8bbe03933f4cf6d0f7f07935aed893380bd25067daa75c92ef450fb0f47f836d` |
| `mouth-E-3.webp` | 42,218 | `cfff6ee50d09b7cebb477cff8a30e4248068a1cb4220a198cb4d83171c4a4cb9` |
| `mouth-E-4.webp` | 42,476 | `22b18d380d465119d152d0070c6c92169aa09443d88634448674ba30fc034068` |
| `mouth-E.webp` | 42,674 | `dfc37ca7ab33431d97bd46d1d2ca5b76f800ca624f25550e88ca1a5ae768d7ea` |
| `mouth-FF.webp` | 41,882 | `eafb2987693190d72042e39b738aa7ad1f6ea48439c26a24e12e1ef22e1ecba3` |
| `mouth-I-1.webp` | 41,450 | `f999d281602bf275bc21f35f0427d486b2dd976f7c6e9ee91d7e22592081c6aa` |
| `mouth-I-2.webp` | 41,514 | `c535d3bad8e986850ef822a542144be68e4942b6bd86e19d568e13ed397de6f4` |
| `mouth-I-3.webp` | 41,754 | `e81244c46670c96eea90e0e3aa67eab02390f4ce6b3aa66cc8d5e63f9e97a349` |
| `mouth-I-4.webp` | 41,838 | `396c3bda9419d7ac13338af98c263aaf7a9c26cce63390ed65844f01223466b1` |
| `mouth-I.webp` | 41,936 | `6c75d8e780926f4d1f8ea662671cd4e5f502fa011465635f7fa8faf1e0ac81b6` |
| `mouth-nn-1.webp` | 42,014 | `59744da879075ee1839284c76566cc3dcfefe9f85ea1757c17de184d6e90fae5` |
| `mouth-nn-2.webp` | 42,266 | `6817d51c379abfbb8852078ac5fa78deacd5eeff07e11a1c4ce338bac29f402a` |
| `mouth-nn-3.webp` | 42,302 | `f6af81f40459c71d54a2e920d13ce7443d112b869a4f816dcd4b40e2c91e4c68` |
| `mouth-nn-4.webp` | 42,564 | `c424567d08d0041317a3d9037abcf0a6ff546613fbd6b7c9fd397f6075cee0c7` |
| `mouth-nn.webp` | 42,802 | `1048b7ce7bd5ea678eda0443b9ad6040ad4e7b8c273dd7fd6c867214fa7f1c33` |
| `mouth-O-1.webp` | 42,332 | `8782302a900c8a68294678460c30446d9febdb323e86cb58ecf76e272b437848` |
| `mouth-O-2.webp` | 42,440 | `a6f9d25e66fc470793db885b86fc9f2cc72c543e0a73b6e878f80e1546267e42` |
| `mouth-O-3.webp` | 42,628 | `cab75b63448028c97b7bfda7af4b3c44d89458b3f4f71469ad343ca29c3e6334` |
| `mouth-O-4.webp` | 42,938 | `a8e93b632e36532a9de2ad3b7984e6ac81f5a2716043ba880ec8910d09833ddf` |
| `mouth-O.webp` | 43,130 | `b96778730f7810aad7a31d1e1e03dc45971229d6ac098b564208b4e51f68f8b0` |
| `mouth-sil.webp` | 40,484 | `48debc923f621a3e0f8cf390df3173119b302640bb0af708fa3b42846710ce74` |
| `mouth-SS-1.webp` | 41,480 | `732bd585c58108dd6e805b73d760ebbabb3fee5d25d4ef59240dad75f911d4c8` |
| `mouth-SS-2.webp` | 41,536 | `76654f6e21b98f42b1518eb974e24827a9145b4271f3b9624517c62dc71d605c` |
| `mouth-SS-3.webp` | 41,640 | `942f6ec2a350635ef236936633a2bd1604dc40ab4a7defccdc6bb851bde752e9` |
| `mouth-SS-4.webp` | 41,830 | `2155874a489cc4fa40c23bbe234b1972e24c00da3fb34ed80118ccac748670eb` |
| `mouth-SS.webp` | 42,070 | `a145403967eb82973f41ac1707bf94576858788009b9d5c6e69a73c35639c201` |
| `mouth-U-1.webp` | 41,828 | `abadb5dac86f2bd2549b1de047e9453a0fabf8b856cae6123852c552bc9eeaa5` |
| `mouth-U-2.webp` | 41,808 | `e3e519631d6fdd4ad7ee95eba20311f61ea151231b5827e888b6fb4850628a3e` |
| `mouth-U-3.webp` | 41,868 | `122bbad6c5ee060564ea651e1d2aeb3c75b06dfd8ead7485786ef589ff504050` |
| `mouth-U-4.webp` | 42,046 | `4a9177b8d69e998c5622fa3228aaf3a08ed8a66a115bcef633f821cf26a5af31` |
| `mouth-U.webp` | 42,356 | `d8a55e1c03d39b00a99c3a60b068f6b0ad98fc96e6ec40c9759ee3e65bf707a4` |

## Terms

This pack is distributed with Heliox OS under the project's own licence, the MIT License (the `LICENSE` file at the
root of the repository). The pictures are generated and show no real person.

OpenAI's Terms of Use (https://openai.com/policies/row-terms-of-use/, effective 1 January 2026), section "Content",
say that as between the user and OpenAI, "you (a) retain your ownership rights in Input and (b) own the Output", and
that OpenAI assigns to the user whatever rights it has in the Output. The same terms forbid presenting Output as made
by a human when it was not; this README and the `credit` line say plainly that the pictures are generated. In several
countries a picture made by an AI without human authorship may carry no copyright at all.
