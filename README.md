# Dependency-Aware RAG for Feature Envy Refactoring

This research implementation investigates whether adding repository dependency
evidence to retrieved context helps a language model refactor C# Feature Envy
without introducing regressions, compared with an otherwise matched retrieval
baseline. The reported study used DeepSeek; this release does not contain model
responses, credentials, or experimental source repositories.

## Architecture

1. Roslyn and MSBuild load a C# project and resolve the target method's symbols.
2. The Python evaluator applies the Feature Envy criteria `ATFD > 5`,
   `LAA < 1/3`, and `FDP <= 5` to resolved fields and properties. Unresolved
   dependencies result in an uncertain classification.
3. Repository methods are retrieved with lexical BM25, local dense embeddings,
   or hybrid reciprocal-rank fusion. The target method is excluded.
4. Matched prompts contain the same task and retrieved methods; the augmented
   condition adds available Roslyn dependency information.
5. Proposals can be validated for patch application, compilation, tests and
   post-change analysis. These outcomes are distinct, and missing evidence is
   not a pass.

The evaluator counts distinct semantic field and property symbols, including
some framework properties and enum members. This operationalization has
construct-validity limitations; the presence or absence of a target-method smell
does not establish that the overall refactoring is safe.

## Reported study

The study used 35 eligible cases, two rounds and three retrieval methods,
yielding 105 matched pairs per round. Across the two rounds, 66 nonempty patches
were applied. Ten applied patches had verified target-level smell resolution.
One newly introduced helper method was metric-positive for Feature Envy. One
patch had a dependency-regression PASS **within a limited, inspected project
scope**; this is not a repository-wide or behavioral safety guarantee.

The evidence does **not** establish superiority of Dependency-Augmented RAG
over Baseline RAG. See [RESULTS.md](RESULTS.md) for counts, denominators, and
important qualification of the evidence.

## Contents

- `src/`: Python orchestration, metrics, retrieval, prompting and validation.
- `csharp-analysis/`: C# Roslyn analyzer, corpus and dependency extraction.
- `tests/`: focused Python unit tests.
- `requirements.txt`: direct Python dependencies, not an experiment lockfile.
- `EVALUATION_EVIDENCE.csv`: patch-level evaluation results for 66 applied patches.

## Using the code

Use Python 3.10 or later. Dependencies are listed in `requirements.txt`; the
exact versions from the experiment and the full transitive dependency lockfile
were not established. Dense retrieval also needs a compatible locally available
sentence-transformer model; no model weights are distributed here. The Roslyn
project targets .NET 5 and declares its package dependencies in its project
file. Its use requires a compatible .NET/MSBuild installation and the referenced
packages. Some bridges invoke the Windows `dotnet.exe` host.

From the release directory, the dependency-free metric unit tests can be run
with:

```powershell
python -m unittest discover -s tests -p test_feature_envy_metrics.py
```

Given a compatible Roslyn analysis JSON file, evaluate the target with:

```powershell
python src\feature_envy_metrics.py --analysis input\analysis.json --output output\metrics.json
```
The release includes aggregate results and patch-level evaluation
classifications for 66 applied patches. However, original C# repositories,
case data, raw model outputs, full patch artifacts, detailed Roslyn evidence,
and the exact experimental environment are not included. Therefore,
the complete experiment cannot be independently reproduced from this
release alone.

The client expects an API credential through an environment variable and makes
live requests only when explicitly enabled. Never commit credentials or raw
request/response artifacts. Running the live pipeline can incur cost and send
repository content to a third party.

