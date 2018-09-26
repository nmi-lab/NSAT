.DEFAULT_GOAL := lib/libnsat.so
# .DEFAULT_GOAL := ./bin/run_any_test
MODE=LB

TARGET = run_any_test
LTARGET = libnsat.so

INCDIR=include
SRCDIR=src
BINDIR=bin
DATADIR=data
OBJDIR=obj
#DEMODIR=tests/python
DEMODIR=tests/c
EXPDIR=examples
LIBDIR=lib
PYTHON=pyNSATlib

CC=gcc
LINKER=gcc -o
# LDFLAGS = -lm -std=c99 -pthread
LDFLAGS = -lm -pthread
#LDFLAGS = -pthread
# LDFLAGS = -lm -std=c99 -pg -pthread -no-pie # profiler
#LDFLAGS = -lm -pthread -pg

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LDFLAGS = -pthread
endif

ifeq ($(MODE), DB)
  	CFLAGS=-g -Wall -Wextra -Wpedantic -fstack-protector-all -fPIC -I$(INCDIR)
	# CFLAGS=-g -Wall -Wextra -Wpedantic -fstack-protector-all -I$(INCDIR)
	# CFLAGS=-g -Wall -pedantic -fstack-protector-all -pg -I$(INCDIR)
else ifeq ($(MODE), LB)
	CFLAGS= -Ofast -ftree-vectorize -fPIC -flto -msse2 -march=native -mtune=native -Wno-unused-result -I$(INCDIR)
else
	CFLAGS=-Ofast -ftree-vectorize -msse2 -pipe -march=native -mtune=native -flto -Wno-unused-result -I$(INCDIR)
	# CFLAGS=-O2 -funroll-loops -flto -I$(INCDIR)
endif

SOURCES  := $(wildcard $(SRCDIR)/*.c)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

DEMOSRC  := $(wildcard $(DEMODIR)/*.c)
#DEMOSRC  := $(DEMODIR)/run_any_test.c
DEMOOBJ  := $(DEMOSRC:$(DEMODIR)/%.c=$(OBJDIR)/%.o)

INCLUDES := $(wildcard $(INCDIR)/*.h)

$(OBJECTS): | $(OBJDIR) $(LIBDIR)

$(OBJDIR):
	mkdir -p $@

$(LIBDIR):
	mkdir -p $@

$(DATADIR):
	mkdir -p $@

$(BINDIR)/$(TARGET): $(OBJECTS) $(DEMOOBJ)
	$(LINKER) $@ $^ $(LDFLAGS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

$(DEMOOBJ): $(OBJDIR)/%.o : $(DEMODIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

$(LIBDIR)/$(LTARGET): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

.PHONY: clean cleanall cleanpyc

cleanpyc:
	@find . -name "*.pyc" -delete

clean: cleanpyc
	rm -rf $(OBJDIR)	\
	rm -f *~ core $(INCDIR)/*~  \
	rm -f $(LIBDIR)/*.so \
	rm -f $(BINDIR)/* \

cleanall: clean
	rm -f $(DATADIR)/*

