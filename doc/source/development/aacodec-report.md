# Archetypal Analysis Codec (AACodec) report

## Workflow

```
                                                                    Blosc
                                                                    Block
                                                                   ┌─────┐           ┌──┬──┐
                                                                   │     │  Celling  │  │  │  AACodec  ┌───┐
   Bidimensional                      Blosc                    ┌─► │     │ ────────► ├──┼──┤ ────────► │   │
   Dataset                            Chunk                    │   │     │           │  │  │           └───┘
┌─────────────────┐            ┌─────┬─────┬─────┐             │   └─────┘           └──┴──┘
│                 │            │     │     │     │             │
│                 │            │     │     │     │             │   ┌─────┐           ┌──┬──┐
│                 │            │     │     │     │             │   │     │  Celling  │  │  │  AACodec  ┌───┐
│                 │            ├─────┼─────┼─────┤  Compress   ├─► │     │ ────────► ├──┼──┤ ────────► │   │
│                 │  Blocking  │     │     │     │  (Parallel) │   │     │           │  │  │           └───┘
│                 │ ─────────► │     │     │     │ ────────────┤   └─────┘           └──┴──┘
│                 │            │     │     │     │             │
│                 │            ├─────┼─────┼─────┤             │      .
│                 │            │     │     │     │             │      .
│                 │            │     │     │     │             │      .
│                 │            │     │     │     │             │
└─────────────────┘            └─────┴─────┴─────┘             │   ┌─────┐           ┌──┬──┐
                                                               │   │     │  Celling  │  │  │  AACodec  ┌───┐
                                                               └─► │     │ ────────► ├──┼──┤ ────────► │   │
                                                                   │     │           │  │  │           └───┘
                                                                   └─────┘           └──┴──┘
```

This codec has been implemented to work with 2-dimensional datasets (for example, images).

First of all, the data is split into 2-dimensional blocks using Blosc.
Since it does not support dimensions, the data have to be rearranged before passing it to Blosc.

After that, the Blosc blocks are compressed in parallel.
However, in order to apply the AACodec, another level of blocking (4x4 cells) have to be performed in each block.
Finally, data is compressed using the AACodec.
```
                                   Archetypes
                                  ┌───┬───┬───┐
┌───┬───┬───┬───┐                 │   │   │   │
│   │   │   │   │             ┌─► │   │   │   ├─────────────────────────┐
│   │   │   │   │             │   └───┴───┴───┘                         │
├───┼───┼───┼───┤             │                                         │
│   │   │   │   │  Archetypal │                                         │
│   │   │   │   │  Analysis   │   ┌──┬──┬──┬──┐                         │   ┌────────────┬─────────┐
├───┼───┼───┼───┤ ────────────┤   │  │  │  │  │                         ├─► │ Archetypes │ T. data │
│   │   │   │   │             │   ├──┼──┼──┼──┤                  ┌┬┬┬┐  │   └────────────┴─────────┘
│   │   │   │   │             │   │  │  │  │  │  float to uint8  ├┼┼┼┤  │
├───┼───┼───┼───┤             └─► ├──┼──┼──┼──┤ ───────────────► ├┼┼┼┤ ─┘
│   │   │   │   │                 │  │  │  │  │                  ├┼┼┼┤
│   │   │   │   │                 ├──┼──┼──┼──┤                  └┴┴┴┘
└───┴───┴───┴───┘                 │  │  │  │  │
                                  └──┴──┴──┴──┘
                                   Transformed
                                   Data
```
## Future work

- Create and register a Blosc filter that implements the celling. 


## Example

In `examples/ex_arch_codec.py` we compress an image using the AACODEC with different 
parameters.
