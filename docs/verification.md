# Source release verification

Local checks for the initial source release:

| Check | Result |
|---|---|
| Imported source coverage | All 59 scripts and launchers present |
| Synthetic recipe and publication tests | 5 passed |
| Mocked planner contract/privacy tests | 5 passed |
| Python compilation | Passed for all source, tools, tests, experiments, and essay tooling |
| MLX training and evaluation CLI startup | Passed with `--help`; no model weights loaded |
| Article build | Passed |
| Numeric figure and architecture diagram builds | Passed; generated assets stay ignored |
| Git index audit | Passed; reviewed hashes match, blocked file types absent |
| Exact local credential and private-identity checks | Passed without printing credential values |
| Local source-literal/conversation overlap check | No ten-word matches found |
| Independent Gitleaks scan of the indexed snapshot | No leaks found |

A scanner initially interpreted a checksum entry as a generic token assignment. The manifest was changed to separate `path` and `sha256` fields. No secret-scanner rule was disabled and no detection was allowlisted.

These checks validate the source publication and its fictional demo. They do not rerun historical private evaluations, train models, validate every upstream model revision, or certify learned weights as private-data-free. GPU and VM branches were syntax-checked; full hardware/model execution remains a separate step with the operator's own inputs.
