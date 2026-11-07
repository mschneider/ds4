#!/usr/bin/env python3
"""Run the tau2 CLI with its NL-assertion judge routed through OpenRouter.

tau2 2.0 grades retail's NL assertions with `gpt-5.4-mini-2026-03-17`
(reasoning_effort low) addressed to OpenAI directly; our runs have only an
OpenRouter key.  This sets the same model under its OpenRouter name
(TAU2_NL_JUDGE overrides) and then runs `tau2` unchanged.  tau2 runs tasks
in threads, so the patch reaches every evaluation.
"""
import os
import sys

import tau2.config
import tau2.evaluator.evaluator_nl_assertions as nl
from tau2.cli import main

judge = os.environ.get("TAU2_NL_JUDGE", "openrouter/openai/gpt-5.4-mini-20260317")
tau2.config.DEFAULT_LLM_NL_ASSERTIONS = judge
nl.DEFAULT_LLM_NL_ASSERTIONS = judge

if __name__ == "__main__":
    sys.argv[0] = "tau2"
    sys.exit(main())
