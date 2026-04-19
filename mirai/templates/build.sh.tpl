#!/bin/bash
set -e

# Auto-detect TensorFlow compile/link flags.
# Override by setting TF_CFLAGS / TF_LFLAGS env vars before running.
if [ -z "$TF_CFLAGS" ]; then
    TF_CFLAGS=$(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))" 2>/dev/null || \
                python2 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))")
fi
if [ -z "$TF_LFLAGS" ]; then
    TF_LFLAGS=$(python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))" 2>/dev/null || \
                python2 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))")
fi

CUDA_INC="${CUDA_INC:-/usr/local/cuda/targets/x86_64-linux/include}"

echo "TF_CFLAGS: ${TF_CFLAGS}"
echo "TF_LFLAGS: ${TF_LFLAGS}"
echo "CUDA_INC:  ${CUDA_INC}"

{% for kernel in kernels %}
echo "Building {{kernel}}.so ..."
g++ -std=c++14 -shared {{kernel}}.cc -o {{kernel}}.so -fPIC ${TF_CFLAGS} -O3 \
    -I${CUDA_INC} \
    -I. \
    ${TF_LFLAGS}
{% endfor %}
echo "Done."
