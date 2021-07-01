CXX = mpicxx
# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -O3 -DPRINT_DIST_STATS -DPRINT_EXTRA_NEDGES -DSTM8_ONESIDED -DUSE_CL_MODEL #-DPRINT_GRAPH_EDGES #-DSET_BATCH_SIZE -DUSE_ALLTOALLV  
# -DPRINT_EXTRA_NEDGES prints extra edges when -p <> is passed to 
#  add extra edges randomly on a generated graph
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS)

OBJ = main.o
TARGET = tric

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	$(LDAPP) $(CXX) $(CXXFLAGS) -o $@ $^ 

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) *.dSYM
