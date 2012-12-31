TARGETS := matrix_simple

all: $(TARGETS)

%: %.c
	tools/compile_executable.sh $@ $< 16 16 1 1

matrix_simple: matrix_simple.c rt/lib.c

.PHONY: clean
clean:
	-rm $(TARGETS) $(addsuffix .ini,$(TARGETS))
