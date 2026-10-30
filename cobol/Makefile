# pyDeltaRCM COBOL kernels - GnuCOBOL 3.2 (apt gnucobol3) on ubuntu-latest.
#
# cobc -C lowers each COBOL program to C; shim.c owns the one-time
# cob_init() runtime startup and the single-op IEEE helpers the kernel
# uses for float arithmetic (see shim.c for why). The result is one
# shared object, ../pyDeltaRCM/lib/pydeltarcm_cobol.so, loaded by
# pyDeltaRCM/cobol_tools.py.

COBC ?= cobc
CC ?= gcc

KERNELS := random_pick.c
# the adapter (pyDeltaRCM/cobol_tools.py) loads the shared object from
# the package directory, so build it there
LIB := ../pyDeltaRCM/lib/pydeltarcm_cobol.so

all: $(LIB)

random_pick.c: random_pick.cob
	$(COBC) -C -O2 random_pick.cob

$(LIB): $(KERNELS) shim.c
	mkdir -p ../pyDeltaRCM/lib
	$(CC) -shared -fPIC -O2 -o $@ shim.c $(KERNELS) \
	    $$(cob-config --cflags) $$(cob-config --libs)

clean:
	rm -f $(KERNELS) *.h $(LIB)

.PHONY: all clean
