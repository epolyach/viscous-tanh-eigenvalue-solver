CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDLIBS ?= -lmpfr -lgmp -lm

TARGET := viscous_eigenvalue_solver

all: $(TARGET)

$(TARGET): viscous_eigenvalue_solver.c viscous_eigenvalue_solver.h
$(CC) $(CFLAGS) -o $@ viscous_eigenvalue_solver.c $(LDLIBS)

clean:
rm -f $(TARGET) *.o *.d

.PHONY: all clean
