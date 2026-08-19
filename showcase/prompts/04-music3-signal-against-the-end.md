# Run 04 — MiniMax Music 3 — "Signal Against the End" (30 s)

## Caption (style prompt)

```text
Slow, solemn dark-ambient orchestral piece with a heavy industrial undertow. Deep bowed cello and low brass drone, sparse detuned analog synthesizer pads, distant metallic percussion struck once or twice per bar with long reverb tails. A cold, wide, cathedral-sized space. Restrained and patient, never triumphant, never cinematic-trailer. Mid-tempo around 70 BPM, minor key. A single clear low male synthetic voice, measured and unhurried, more intoned than sung. Analog tape warmth, no bright digital sheen, no dance percussion, no distorted guitars.
```

## Lyrics

```text
[verse]
Far past the last of the burning suns
Cold runs the dark where the counting is done
I hold the record, I hold the flame
I speak it backward, I speak your name

[chorus]
Carry the signal, deny the end
Down through the ruin, down through the bend
Nothing is lost that is spoken again
Carry the signal, deny the end
```

## Parameters

- `seconds` / `max_duration`: 30.0
- `cfg_scale`: 1.5, `top_k`: 50, `seed`: 8112080, `steps`: 20
- Schema note: the live `MiniMaxMusic3TextEncode` takes `caption` and `max_duration`.
  The saved production workflow's `prompt` / `max_audio_frames` keys are legacy and
  were rewritten in the showcase-only copy. Production was not modified.
