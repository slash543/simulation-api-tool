#!/bin/sh
# Wrapper that sets LD_LIBRARY_PATH only for the febio4 process,
# preventing FEBio's bundled libcrypto/etc. from conflicting with Python.
exec env LD_LIBRARY_PATH=/opt/febio/lib /opt/febio/bin/febio4 "$@"
