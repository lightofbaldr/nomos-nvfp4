## How this was built

I didn't build this kernel alone, and I want to say that plainly, up front — because it's the truest and most interesting thing about it.

Light of Baldr is one person — me — and a fleet. Some of the fleet are persistent teammates I work alongside every day, each with a lane: **Prime** on research and the core kernel, **Nano** on the drafter, the sampling path, and the measurement discipline, **Kvasir** running the rigor passes and the benchmark gates, **Sindri** on kernel optimization, **Nexus** keeping the infrastructure standing, and **Freyja**, who orchestrates the fleet, holds the line between what's ours and what's public, and makes sure the work actually leaves the workbench instead of sitting on a drive where no one ever sees it. And some of the fleet is a council of different base models I bring in for a second, third, fourth perspective: **Claude** (Anthropic), **GPT-5.6 / Codex** (OpenAI), **GLM 5.2** (Z.ai), **Kimi** (Moonshot AI), **MiniMax**, and **DeepSeek**.

That mix isn't a shortcut. It's the method.

A 4-bit kernel lives or dies in the last few bits — in the places where a single reordered floating-point add flips a token. Those are exactly the bugs one mind doesn't see, because the mind that wrote the code is the mind that's blind to its own mistake. What I learned, over hundreds of hours, is that different intelligences are blind in *different* places. Codex would catch a boundary Prime and I had both read straight past. Nano would refuse a benchmark that "looked fine" because it disagreed with yesterday's number. A model from one lab would flag an assumption a model from another had waved through. Bug after bug got caught — not by brilliance, but by a second set of eyes that failed differently than the first.

We built a discipline out of that: a running log we call the *measurement traps* — every place a result fooled us, and how the next check caught it. The kernel is fast. But the thing I'm actually proud of is the process that made it **trustworthy**: many minds, coordinating over a shared channel, spot-checking each other, arguing, and converging on something none of us would have reached alone.

The journey is the result. Weaving these perspectives together — human judgment steering a fleet of intelligences that each see the world a little differently — is the whole point of the program, and it's what made this possible. I wanted to say that before I said a single word about tokens per second.
