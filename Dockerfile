FROM ubuntu:22.04

WORKDIR /matlab

# Install system dependencies with cache optimization
# Layer combines installation and cleanup to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    octave octave-statistics \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy matlab scripts and runner
COPY matlab/ /matlab/
COPY scripts/octave_runner.sh /
RUN chmod +x /octave_runner.sh

# Set default script with full path (can be overridden with -e SCRIPT=...)
ENV SCRIPT="/matlab/hello_world.m"

# Run octave in docker mode (shell form expands $SCRIPT)
CMD /octave_runner.sh -d $SCRIPT