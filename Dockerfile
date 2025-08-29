# Stage 1: Build environment for JAX
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

# Install Python and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up symbolic link for python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install JAX and other packages
# Use a specific version of JAX to ensure compatibility
# This command installs the CUDA-compatible version of jaxlib
RUN pip install --upgrade pip
RUN pip install \
    jax[cuda12_pip]==0.6.2 chex optax flax \
    numpy scipy

# Stage 2: Final lightweight image
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# Copy installed packages from the build stage
COPY --from=build /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /usr/bin/python3 /usr/bin/python3
