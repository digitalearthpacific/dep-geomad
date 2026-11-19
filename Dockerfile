FROM ghcr.io/osgeo/gdal:ubuntu-full-3.7.3

# Don't use old pygeos
ENV USE_PYGEOS=0

RUN rm -f /etc/apt/sources.list.d/apache-arrow.sources \
    && apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libpq-dev \
    ca-certificates \
    build-essential \
    # odc-algo now requires rust
    rustc \
    cargo \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN pip3 install --upgrade pip setuptools wheel
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ADD . /code

WORKDIR /code

# Provenance, recorded in output STAC Items by dep_tools (processing extension).
# Supplied at build time: --build-arg DOCKER_IMAGE=<image:tag> --build-arg COMMIT=<sha>
ARG DOCKER_IMAGE
ARG COMMIT
ENV IMAGE_TAG=$DOCKER_IMAGE
ENV COMMIT=$COMMIT
ENV IMAGE_REPO=https://github.com/digitalearthpacific/dep-geomad/tree/$COMMIT/
