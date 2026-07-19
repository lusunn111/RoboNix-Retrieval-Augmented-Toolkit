# Architecture Image Prompt

The current README asset was generated from the prompt below and saved as
`retrieval-memory-overview-v2.png`. Reuse this prompt when regenerating or
iterating on the architecture figure.

## Web ImageGen prompt

```text
Create a beautiful, premium scientific architecture infographic for a top-tier embodied-AI open-source GitHub repository.

Canvas and visual language:
- Wide 16:9 landscape composition, at least 1920x1080.
- Clean white background with subtle pale-lavender and pale-teal technical textures, generous whitespace, and no decorative clutter.
- Modern 2.5D vector-like rendering, rounded module cards, crisp arrows, restrained deep-purple, indigo, and teal palette.
- The experience memory should look like a sophisticated vector database or memory crystal, while remaining technically credible.
- Publication-ready visual hierarchy comparable to a flagship AI research repository, not a generic office flowchart.

Title, rendered exactly:
"ROBONIX RETRIEVAL-AUGMENTED MEMORY"

Top lane title, rendered exactly:
"OFFLINE EXPERIENCE MEMORY"

Top lane flow from left to right:
"Robot History" -> "Two-View Encoder" -> "4,352D Embedding" -> "Qdrant Experience Memory"

Under "Two-View Encoder", add two small camera inputs labeled exactly "Third-Person" and "Wrist View". Add two compact encoder tags labeled exactly "DINOv2" and "SigLIP".

Bottom lane title, rendered exactly:
"ONLINE EXPERIENCE RETRIEVAL"

Bottom lane flow from left to right:
"Current Views + Task" -> "Embedding Service" -> "Top-k Retrieval" -> "Action Trajectory Candidates" -> "Safety Check" -> "Robot Execution"

Draw a clear fallback arrow from "Safety Check" to a module labeled "Target VLA Policy". Draw a thin update loop from completed robot experience back to "Qdrant Experience Memory", labeled exactly "UPDATE · DEDUP · CLEAN".

Add one compact service ribbon at the bottom, rendered exactly:
"HeiSD · RT-Cache · Qdrant · Custom Encoder"

All modules and arrows must stay fully inside the frame. Use large, readable sans-serif typography. Render every supplied label verbatim. Do not add extra text, equations, code, people, company logos, watermarks, dark backgrounds, excessive gradients, duplicate modules, or cropped edges.
```

## Review checklist

- Every label is spelled exactly as provided, including `4,352D` and `Qdrant`.
- Offline database construction and online retrieval are visually distinct.
- The update loop and target-policy fallback are both clear but secondary.
- The complete diagram remains readable when displayed at 80% README width.
