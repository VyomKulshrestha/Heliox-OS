# Portrait, man (`portrait-man`)

A Heliox face pack (docs/FACE_PACKS.md, format 1): a photographic portrait of a person who does not exist, with
mouth pictures for speech, blink pictures and thinking eyes.

Generated with ChatGPT image generation in November 2026. Shows no real person.

## Where the pictures come from

Every source picture was generated with ChatGPT image generation (OpenAI) on 19 November 2026. The page showed no
model name. First an original portrait was made from this prompt, with DESCRIPTION "European man in his early thirties" and
the hair sentence "Short neat hair, clean-shaven, so the whole face, the jawline and the hairline are clear.":

> Create a photorealistic studio portrait photograph of a fictional [DESCRIPTION] who does not resemble any real person or celebrity. Head and shoulders, facing the camera straight on, head level and centred, eyes open and looking into the lens, a neutral relaxed expression with the lips gently closed. [HAIR SENTENCE] Soft even light from the front with a faint cool rim light, a plain very dark charcoal background, a plain dark crew-neck top, no jewellery, no glasses, no text, no watermark, no hands. Natural skin texture, sharp focus. Square image, the face filling about half the height of the frame.

Every other picture is an edit of that original, made by attaching it and sending this prompt with CHANGE filled in
(for the blink pictures without the words "eyes open and looking at the camera"; a few early edits used the same
prompt without them as well):

> Make one edit to this portrait. Keep everything else exactly the same so the new image can be laid directly over the original: the same person, the same framing and head position, the same lighting, hair, clothes and background, the same image size, eyes open and looking at the camera. The only change: CHANGE.

No picture was retouched by hand.

## Source pictures used

| File | Used for | CHANGE | SHA-256 |
|---|---|---|---|
| `man-2.png` | the idle picture and the closed mouth (`sil`) | the original portrait | `ab5097ad1e2e00a445e2fa25439cb678ac7021cc313ffba6e226b9bf21139c27` |
| `man-2-mouth-m.png` | mouth `PP` | the lips are pressed gently together as if humming 'm' | `53434091d350b1db74c8bb799d3be147ea4ad40ecc3b3cf0a9433eea340c296e` |
| `man-2-mouth-f.png` | mouth `FF` | the upper teeth rest on the lower lip as if saying 'f' (sent in the same chat without attaching the original again, so the page edited the portrait from earlier in that chat; the result differs from the original by about 2.7 levels of 255 on average, like the other edits) | `8d1b56aae63905241a17429287e115be1700d51ceccb3bc5d924b98f3e9485fe` |
| `man-2-mouth-s.png` | mouth `SS` | the teeth are almost closed and the lips slightly parted and spread as if saying 's' | `f3a7689402d55d8b6e9290a6e5dbd0c2463d8d7cbaefa40a8d04ebb85e1f4942` |
| `man-2-mouth-ah-soft.png` | mouth `aa` | the mouth is open only a little, as in the middle of relaxed everyday speech, lips apart by about a finger's width, the upper teeth barely visible | `ee6786eddf3b06d1d50f0fe6a9feab250570d06a1c6a254c3265dca69003c93e` |
| `man-2-mouth-eh.png` | mouth `E` | the mouth is a little open and relaxed, as in relaxed everyday speech on the 'e' of the word 'bed', teeth slightly apart | `c945489b86968c631c844bba4d77f2baa762b137744bf6acc40a947e10ce3140` |
| `man-2-mouth-ee-soft.png` | mouth `I` | the lips are slightly spread and barely parted, as in the middle of relaxed everyday speech on an 'ee' sound, only a hint of the upper teeth, not a smile and not a grin | `f94bd207df829fe9d8266f3ca466b824b0cecf134621480624d02268b94a43b6` |
| `man-2-mouth-oh-soft.png` | mouth `O` | the lips are gently rounded and a little open, as in relaxed everyday speech on an 'oh' sound | `731ab5886ac6a68a8f9b9c8ba3617ab816da99137c11f8b41e14a3af1b2be2e8` |
| `man-2-mouth-oo-soft.png` | mouth `U` | the lips are gently rounded and nearly closed, as in relaxed everyday speech on an 'oo' sound, not a pucker and not a kiss | `71c60d394eaf4d890298958abd5ac8ce3ebda62ef90094e49c7a2468cf03c754` |
| `man-2-eyes-closed.png` | blink (closed) | his eyes are gently closed, as in the middle of a relaxed blink | `d4e1bca4797e60a987174c9105b95cde33040589bc007f817d38fc980d792d4f` |
| `man-2-eyes-half.png` | blink (half) | the eyelids are half lowered, as in the middle of a blink, with the eyes still partly visible and the lips closed | `7ee69813ebee518af7b5bc9abbbcb8d670ad58e7e8edf2c2c0674a17eac7c9fc` |
| `man-2-thinking.png` | thinking eyes and the glance | the eyes look up and a little to one side, as if thinking, with the lips still closed | `957c3124bffd1f16617a2c54bb6bfe04a21e4d95111dbad734ccac647dee7072` |

## How the pack was made

Processor-only image work with OpenCV (no neural network beyond OpenCV's small YuNet face detector, used only to
place the rectangles). Each edited picture was aligned to the original on the forehead, nose and cheeks, and the
head was cut to a 512 x 512 square (`idle-00.webp`, WebP quality 80).

Each mouth picture covers the whole lower face, `mouth.rect` [129, 280, 278, 214], from the nose to below the chin,
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

- `PP`: the lips and the area around them come from the edited picture itself, so the idle picture's fuller lips never show under the pressed lip line.
- `E` is carried 1.45 times as far as its edited picture shows (its whole motion), so the shape reads at the small size the face is shown at.
- `I` is carried 1.80 times as far as its edited picture shows (its motion, the lips' own movement included), so the shape reads at the small size the face is shown at.
- `mouth.chin` gives each shape's chin travel at the full shape in pack pixels, down positive.
- `FF` has no in-between pictures: teeth meeting the lower lip did not morph cleanly.
- Shapes without a picture of their own (the app shows the closed mouth or the nearest shape the pack has): `TH`, `DD`, `kk`, `CH`, `nn`, `RR`.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `blink-closed.webp` | 5,442 | `87e43585e03119b751d6d9c06cdd910fbbc2df19b1d553abc8629ba2e877e104` |
| `blink-half.webp` | 5,484 | `e4551abdfbaa741aba63210648d5639b3130cfe84ebdd46511487566d2b540ab` |
| `eyes-glance-1.webp` | 5,578 | `0fdef18e4a876aaada76fce03af1d1efc49f21299068c8837ee25475ab672d39` |
| `eyes-glance-2.webp` | 5,526 | `297a5343fda8cad91c0884a7b2f591c2283c99d8d7c949932e8d68a8adae7ae5` |
| `eyes-thinking.webp` | 5,680 | `02486e74bace2359c0d905c23b372ee9bd11d3529e2e31dd67426eaac002a016` |
| `idle-00.webp` | 13,552 | `00d454129ce55959e6ffcc8a3f8ba93c20eac77f075e564425ff49dd941cb536` |
| `manifest.json` | 1,911 | `468753bda836c56f550d5861c50e225ada4159a12c9a5f27bf32881ebf5cf52e` |
| `mouth-aa-1.webp` | 44,154 | `7673a9fd8e604a47f5cdd9ce62a406ead9c0c669b7a2a1a78e8fa56aa50d27be` |
| `mouth-aa-2.webp` | 44,328 | `7e89a4a68c5a49d8432b9ae0742050a5ffd74de77a459f0e9dd46a8f183a9f85` |
| `mouth-aa-3.webp` | 44,440 | `49649c61efe1d38b86b786e05fbbb0b9752ec58f477b8e643454f8f6d8ffe239` |
| `mouth-aa-4.webp` | 44,744 | `210db8697df0ecc662f5f903eb8864d50b95bbc4f24782cf3624fbe4694e587c` |
| `mouth-aa.webp` | 44,814 | `93d08ea604c1176e18464fa404465259134244e4d46c556f365bd0e5b986e140` |
| `mouth-E-1.webp` | 43,922 | `811f7326da64b9164c7c967a99211969aca436e7fb21f0cae4f5e0cf808b9976` |
| `mouth-E-2.webp` | 44,194 | `e84b09bcd65a91408ae92bc737e60b0d3e85f1666b5885a9c514541b9e57a789` |
| `mouth-E-3.webp` | 44,400 | `44cfe00f17822fd5fdd14139d4e2b66084f8b0701bdd9199233284eb5f3137dc` |
| `mouth-E-4.webp` | 44,530 | `96c3b6fb50ca012cd7a93e2bb96526c0bb6aacd6dc9e651670e856ce2987249f` |
| `mouth-E.webp` | 44,714 | `65934a5c5ac2d3e8956fe963387b6e39e42c6d43ef692896ee1fdae3e591d0e1` |
| `mouth-FF.webp` | 43,718 | `64c79e6276bf265eb73e5ffce3ff53d861ed9e522d43454849d6ff066740c68b` |
| `mouth-I-1.webp` | 44,126 | `163d4944f487be9f04a3837d21ab9ef0d74edfb771de44bbde985913772a510f` |
| `mouth-I-2.webp` | 44,134 | `625f2e1e256898028331936a2f15169940993d85118115476f040d993b752639` |
| `mouth-I.webp` | 44,488 | `0bfff89f7e77bc2be8f64180810bbf34aef5985db9d1ce7130988d1930ea7bb1` |
| `mouth-O-1.webp` | 44,156 | `9bd2170534934797bb97035541857e24ef210f274da13380dd5c3d95f8c289ca` |
| `mouth-O-2.webp` | 44,316 | `1bb8cd4b675810780913b169b241194278fc599913fef263fd20b219a17db746` |
| `mouth-O-3.webp` | 44,524 | `226c447ff0a61207d2af52216c781f5794eefd96ca63fb5bfc01c584182d136d` |
| `mouth-O-4.webp` | 44,724 | `fcf34388bf2991cae75b62be09703341e3ff019364b9bd32b85e540d7a7e9363` |
| `mouth-O.webp` | 45,070 | `019042becb164ff432fcb335405d1aa3d1f3ee225e3b90203dd9357b60dd3b02` |
| `mouth-PP-1.webp` | 43,660 | `6d6781f029c7c8394843ce1adaa7e274fe9772f85515c8adb9b7e6d2ca65ba07` |
| `mouth-PP-2.webp` | 43,840 | `58bfc2e581d5e4af44239cbe1e409890d07bd4fded05c8dccdccca564764383c` |
| `mouth-PP-3.webp` | 43,724 | `de5d4007c7ce59d983887f05c3ee3920dfd49021afd19977f94eb5737c252aa5` |
| `mouth-PP-4.webp` | 44,140 | `bc10ec577ee3e2eb634067fa1bbde78c304294543d2304665550a02e49f3fe44` |
| `mouth-PP.webp` | 44,390 | `e4966d99aae25add81cfac0e3b2f2543c5a2057abe081f9263b2b4c39de71452` |
| `mouth-sil.webp` | 42,150 | `af3089f4127a4744734b17e53bd17e52532afb2945a0e26df067c0e274400ef3` |
| `mouth-SS-1.webp` | 43,544 | `6ca9cf8bf06e37f5d2bff029ba9eae9e2d96f8f411ba99a31c1b34fc383dcb9d` |
| `mouth-SS-2.webp` | 43,712 | `8c98c49798855153ea32b04e78613a2e614be3828f43af678ebdfa42e629f0c7` |
| `mouth-SS-3.webp` | 43,758 | `e19d0d29b79906ce6c94af3ef8b206d4935a82e8cf05d8a3b396717a9d6bbde2` |
| `mouth-SS-4.webp` | 44,068 | `edec6a331734c331d7dd3a3b5e9e47d6008e1b607d785d830fff4ac92c9c3cb7` |
| `mouth-SS.webp` | 44,170 | `d09784029122cf2d837f447b21b8dfae000a6383536571dcbe3d7ba4098a3205` |
| `mouth-U-1.webp` | 44,020 | `984bfeba5eaf9c32c3d95fb8c550d73cb70e2bb30edca3418440c1e21bf086f0` |
| `mouth-U-2.webp` | 44,216 | `e2d40cac09e62b9e0587086f3f3aefcdc555c27e04b1dcadc120d464576e4507` |
| `mouth-U.webp` | 44,456 | `c72fea17e843d626f56713673e36d351bae20e3b8a52579ee8ba0ac730e781b6` |

## Terms

This pack is distributed with Heliox OS under the project's own licence, the MIT License (the `LICENSE` file at the
root of the repository). The pictures are generated and show no real person.

OpenAI's Terms of Use (https://openai.com/policies/row-terms-of-use/, effective 1 January 2026), section "Content",
say that as between the user and OpenAI, "you (a) retain your ownership rights in Input and (b) own the Output", and
that OpenAI assigns to the user whatever rights it has in the Output. The same terms forbid presenting Output as made
by a human when it was not; this README and the `credit` line say plainly that the pictures are generated. In several
countries a picture made by an AI without human authorship may carry no copyright at all.
