# Sandbox image for running model-written Python and collecting what it produces.
#
# There is no service in here. The control plane starts ONE CONTAINER PER RUN with
# `--network none` and a bind-mounted working directory, and reads the exit code, the
# output streams and the files straight back. See ui/backend/app/sandbox.py.
#
# Why per-run rather than a long-lived service: a service has to be reachable, and on
# Docker Desktop (WSL2) a reachable container also has working outbound networking --
# `--internal` breaks published ports, and disabling bridge masquerade is bypassed by the
# VM's own NAT (both measured). `--network none` is the only arrangement that verifiably
# blocks egress, and it rules out a listening socket by construction.
#
# The library list is chosen for what chat is actually asked to make -- charts,
# spreadsheets, documents, CSVs -- so that generated code works on the first try instead of
# failing on a missing import the user then has to diagnose.
FROM python:3.12-slim

# fonts-dejavu-core: without a font matplotlib renders every label as a box.
# libgomp1: numpy/scipy OpenMP kernels at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         fonts-dejavu-core libgomp1 \
         libxrender1 libxext6 libx11-6 libexpat1 \
    && rm -rf /var/lib/apt/lists/*
# libxrender1/libxext6/libx11-6/libexpat1: rdkit's drawing module (rdMolDraw2D) links against
# them even when rendering headlessly to PNG, so `import rdkit` works but
# `from rdkit.Chem import Draw` fails with "libXrender.so.1: cannot open shared object file".
# Found with `ldd rdMolDraw2D*.so | grep 'not found'` -- check that way if another appears.

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Non-root. The uid is fixed so files written into the bind-mounted run directory have
# predictable ownership.
RUN useradd --create-home --uid 10001 sandbox

# Bake the matplotlib font cache into the image at a fixed MPLCONFIGDIR. Every run is a
# fresh container, so without this EVERY chart -- not just the first -- would pay several
# seconds rebuilding the font list. Owned by the sandbox user so matplotlib finds it
# writable and uses it as-is rather than falling back to a temp dir and rebuilding.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/opt/mplconfig
RUN mkdir -p /opt/mplconfig \
    && chown -R sandbox:sandbox /opt/mplconfig \
    && MPLCONFIGDIR=/opt/mplconfig python -c "import matplotlib.pyplot as plt; plt.figure()" \
    && chown -R sandbox:sandbox /opt/mplconfig

# Scripts run through this rather than directly: it redirects plt.show() (a silent no-op
# without a display, which discards the chart) to a PNG, and writes out any figure still
# open when the script ends.
COPY bootstrap.py /opt/ft/bootstrap.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER sandbox
WORKDIR /work

# The control plane supplies the command (python -I script.py). This default just makes
# `docker run freeswarm-sandbox:latest` self-describing.
CMD ["python", "-c", "import sys; print('FreeToken sandbox image, Python', sys.version.split()[0])"]
