CXX = g++
DBGFLAGS = -g -fomit-frame-pointer
OPTFLAGS = $(DBGFLAGS) -O3
SNTFLAGS = -std=c++11 -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 $(OPTFLAGS) -I.

OBJ = MurmurHash3.o
TARGET = MurmurHash3.a

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(TARGET):  $(OBJ)
	ar rcs MurmurHash3.a $(OBJ) 

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET) *.dSYM
