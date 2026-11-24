# hello-mlsys-agent tool image: the simulator, the lab library, the teacher's prompts and rubrics, and the
# calibration gate, for running jobs (python -m mlab.jobs ...) in a container or on a cluster job runner.
# No coding agent or Node toolchain is included; the student's agent runs elsewhere.
#
#   docker run --rm -i ghcr.io/mlsys-io/hello-mlsys-agent:<tag> simulate < payload.json
FROM python:3.12-slim-bookworm@sha256:82836454a795d9d75549a9e157fd08dbac6c383292cf89e99f4248a97ce6b2ff

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/hello-mlsys-agent

# The lab finds its tasks, templates and catalogs relative to its source tree, so both packages are installed
# editable from this checkout rather than as wheels.
COPY LICENSE README.md ./
COPY packages/simulator packages/simulator
COPY packages/lab packages/lab
COPY scripts/calibration scripts/calibration
COPY calibration/budgets.yaml calibration/budgets.yaml
RUN pip install -e packages/simulator -e packages/lab \
 && useradd --create-home --uid 10001 mlab \
 && python -m mlab.jobs list

# No USER: job runners such as FlowMesh start the container as root to stage inputs and the output directory,
# then drop to an unprivileged uid themselves (and replace the entrypoint). Standalone, prefer
# `docker run --user mlab --network none ...`; the jobs need neither root nor a network.
ENTRYPOINT ["python", "-m", "mlab.jobs"]
CMD ["list"]

LABEL org.opencontainers.image.source="https://github.com/mlsys-io/hello-mlsys-agent" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="mlsys-simulator and mlab job entrypoints (mlab.jobs.v1)"
