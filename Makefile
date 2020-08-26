CC = gcc
CXX = g++
RM = rm -f

CXXFLAGS= 
CXXFLAGS+= `pkg-config --cflags eigen3`
CXXFLAGS+= -std=c++11 -O3 -ffast-math -funsafe-math-optimizations -fopenmp #-static
CXXFLAGS+= -DNDEBUG

LDFLAGS=  -lm

SRCS=simple_example.cpp advanced_example.cpp 

OBJS=$(SRCS:.cpp=.o)

MAIN=advanced_example simple_example

.PHONY: depend clean

all:    $(MAIN)

advanced_example : advanced_example.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

simple_example : simple_example.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)


.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@


depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS) $(MAIN) .depend

include .depend

