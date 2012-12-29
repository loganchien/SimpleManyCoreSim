TARGETS := matrix_simple

all: $(TARGETS)

%: %.c
	arm-none-eabi-gcc -mthumb -Bstatic -static $^ -include rt/lib.h -o $@

matrix_simple: matrix_simple.c rt/lib.c

.PHONY: clean
clean:
	-rm $(TARGETS)
