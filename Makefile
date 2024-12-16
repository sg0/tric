CXX = mpicxx
DBGFLAGS = -g -fomit-frame-pointer
OPTFLAGS = $(DBGFLAGS) -O3 -DPRINT_EXTRA_NEDGES #-DAGGR_BUFR_INRECV #-DDEBUG_PRINTF #-DDOUBLE_RECV_BUFFER #-DAGGR_HASH2 #-DAGGR_PUSH #-DAGGR_HEUR -DAGGR_PUSH -DAGGR_HASH -DAGGR_BUFR#-DDEBUG_PRINTF #-DAGGR_MAP -DUSE_STD_UNO_MAP_MAP #-DUSE_STD_MAP #-DAGGR_HEUR -DAGGR_MAP -DDEBUG_PRINTF -DAGGR_BUFR -DAGGR_BUFR_RMA#-DDEBUG_PRINTF   
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++20 -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++20 $(OPTFLAGS) -I.
LDFLAGS = ./murmurhash/MurmurHash3.a

OBJ = main.o
TARGET = tric

ENABLE_OPENMP=0
ifeq ($(ENABLE_OPENMP),1)
CXXFLAGS += -fopenmp -DUSE_OPENMP
endif

ENABLE_RAPID_FAM=0
ifeq ($(ENABLE_RAPID_FAM),1)
RAPID_ROOT = /share/micron/rapid/install/gcc-release
CXXFLAGS += -DUSE_RAPID_FAM_ALLOC -I$(RAPID_ROOT)/include
LDFLAGS += -Wl,-rpath=$(RAPID_ROOT)/lib64 -L$(RAPID_ROOT)/lib64 -lrapid 
endif

ENABLE_TAUPROF=0
ifeq ($(ENABLE_TAUPROF),1)
TAU=/people/ghos167/builds/tau-2.32/x86_64/lib
CXX = tau_cxx.sh -tau_makefile=$(TAU)/Makefile.tau-mpi-openmp-mpit
endif


all: $(TARGET)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^
	cd ./murmurhash && $(MAKE)

$(TARGET):  $(OBJ)
	$(LDAPP) $(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) 

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) *.dSYM
