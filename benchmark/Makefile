TARGETS := matrix_simple_16 matrix_simple_256 helloworld

all: $(TARGETS)

matrix_simple_16: matrix_simple.c rt/lib.c
	tools/compile_executable.sh $@ $< 16 16 1 1 -DSIZE=16

matrix_simple_256: matrix_simple.c rt/lib.c
	tools/compile_executable.sh $@ $< 16 16 16 16 -DSIZE=256

helloworld: helloworld.c rt/lib.c
	tools/compile_executable.sh $@ $< 4 4 4 4

.PHONY: clean
clean:
	-rm $(TARGETS) $(addsuffix .ini,$(TARGETS))
