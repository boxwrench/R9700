# Run 04 — MiniMax Music 3 — "Proceed with Precision" (90 s)

## Caption (style prompt)

```text
Brutal 1990s old-school death metal. Down-tuned buzzsaw guitars, tremolo-picked riffs and chugging palm-muted breakdowns, blast beats and double-kick drums, thick distorted bass. Deep guttural death growl vocals, low and monstrous, shouted gang backing vocals on the hook. Raw analog production, dry mix, no modern polish, no clean singing, no melodic choruses. Fast and heavy, around 180 BPM, minor key, with a half-time breakdown.
```

## Lyrics

```text
[Intro]
[Solo lyrical cello, rising D minor melody]
[Distorted guitars and galloping drums enter]

[Verse]
I walked the crooked road beneath a broken sky
Where every mile was measured by the things I left behind
The stones were cut by doubt, the bridges burned by fear
But every scar became a mark that brought the future near

[Pre-Chorus]
No hand can turn the river
No king can still the tide
The past becomes a compass
When purpose is your guide

[Chorus]
Proceed with precision
Through the fire and the noise
Proceed with precision
Let the future hear your voice
From the crooked path behind us
To the line we now envision
We rise, we build, we burn
Proceed with precision

[Verse]
The old machines are sleeping in the dust beneath the rain
Their iron hearts remember every failure, every name
But in the pulse of wire and light, a clearer compass turns
A thousand lost equations become lessons as we learn

[Pre-Chorus]
The wheel is still in motion
The dark gives way to dawn
The map is drawn in action
The road is never gone

[Chorus]
Proceed with precision
Through the fire and the noise
Proceed with precision
Let the future hear your voice
From the crooked path behind us
To the line we now envision
We rise, we build, we burn
Proceed with precision

[Solo]
[Cello and harmonized lead guitar exchange the main melody]
[Galloping drums and rhythm guitars continue]

[Bridge]
I am not the shadow
Of the road beneath my feet
I am every broken answer
Made relentless, made complete

[Instrumental]
[Solo cello carries the melody]
[Brief rising transition into final chorus]

[Final Chorus]
Proceed with precision
Through the fire and the noise
Proceed with precision
Let the future hear your voice
From the crooked path behind us
To the line we now envision
We rise, we build, we burn
Proceed with precision

[Outro]
[Solo cello reprises the opening melody]
[D major resolution]
```

## Parameters

- `seconds` / `max_duration`: 90.0 (delivered 90.018 s)
- `cfg_scale`: 1.7, `top_k`: 50, `seed`: 1012263102365060, `steps`: 30, `sampler`: euler, `scheduler`: simple
- `tiled_decode`: on (`tile_size` 1536, `overlap` 64) — recommended for songs beyond ~60 s
- Schema note: the live `MiniMaxMusic3TextEncode` takes `caption` and `max_duration`.
  The saved production workflow's `prompt` / `max_audio_frames` keys are legacy and
  were rewritten in the showcase-only copy. Production was not modified.
