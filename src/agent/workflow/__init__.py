"""Ultracode: a big task as a planned, checked, resumable workflow.

    plan.py         the plan: steps, dependencies, checks; parsing and validating what a model wrote
    roles.py        what each role (planner, worker, critic, advisor) is told
    store.py        a run on disk (`.argent/workflows/<id>.json`): survives a crash, can be resumed
    calibration.py  how much of the machinery this model on this PC can carry (the ladder 3/2/1/0)
    advisor.py      the stronger second opinion at the decision points (cloud or local, set by the person)
    engine.py       the loop that runs a plan; knows nothing about providers, processes or the screen
    runtime.py      the real parts the engine is plugged into, and the session switch
    report.py       what the person (and the model that started it) is told at the end

The engine is the only part with logic worth testing hard, so it talks to the world through `Ports`
(plain callables) and the tests plug in stand-ins.
"""
