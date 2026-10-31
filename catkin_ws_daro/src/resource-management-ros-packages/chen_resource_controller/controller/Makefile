CXX=g++
CC=gcc
CFLAGS=-c -O3 -g
CXXFLAGS=-c -O3 -std=c++17 -g
LDFLAGS=-lm -lcgroup -pthread `pkg-config libusb-1.0 --libs` 

CXX_SRC=$(wildcard *.cpp)
C_SRC=$(wildcard *.c)
CXX_OBJ=$(CXX_SRC:.cpp=.o)
C_OBJ=$(C_SRC:.c=.o)
CONTROLLER_EXECUTABLE=controller

DEPS=$(CXX_SRC:%.cpp=%.d)
DEPs+=$(C_SRC:%.c=%.d)

all: $(CONTROLLER_EXECUTABLE)

$(CONTROLLER_EXECUTABLE): $(CXX_OBJ) $(C_OBJ)  
	$(CXX) $(CXX_OBJ) $(C_OBJ) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP $< -o $@

clean:
	rm -rf *.o *.d $(CONTROLLER_EXECUTABLE)
	
-include $(DEPS)
