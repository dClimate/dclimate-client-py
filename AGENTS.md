# AGENTS.md

## Cross-client parity

This Python client and [dClimate/dclimate-client-js](https://github.com/dClimate/dclimate-client-js) are sibling libraries. Keep their user-visible capabilities and behavior aligned unless a language or runtime difference makes a change inapplicable.

- For every public API or behavior change—especially STAC/IPFS resolution, dataset loading and selection, metadata, errors, and catalog listing—inspect the corresponding implementation, tests, documentation, and relevant open work in the JavaScript client before finishing.
- Unless the user explicitly limits the task to one repository, treat an applicable sibling-library update as part of the same task. Add equivalent tests and documentation in both projects, using idiomatic APIs for each language rather than mechanically copying implementation details.
- If a change is not applicable to the sibling, or the sibling cannot be updated in the current task, state the reason and leave a concrete follow-up in the handoff or pull-request description. Do not silently allow accidental divergence.
- When reviewing either client, treat undocumented behavioral differences as possible defects and verify whether parity should be restored.
