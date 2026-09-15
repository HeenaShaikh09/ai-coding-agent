# agent-sandbox: isolated execution environment for the AI coding agent.
# Runs with --network none at execution time, so any library the agent's
# generated code might need must be baked in HERE at build time - there's
# no internet access available once a task is actually running.

FROM python:3.12-slim

WORKDIR /workspace

# Common libraries likely needed for coding tasks / hackathon-style data
# problems (CSV handling, basic data analysis, HTTP client code that
# won't actually be able to reach the network at runtime, math/plotting).
# Keep this list deliberately small - every library added here makes the
# image bigger and slower to build/rebuild.
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    matplotlib \
    python-dateutil

# No CMD/ENTRYPOINT needed - the agent's run_python() passes the exact
# command ("python filename.py") when it runs the container.
