# Makefile for Genesis GPU batch pipeline

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I../../../Nova
LDFLAGS = -L../../../Nova/build -lnova_compute -lvulkan -lpthread

# Target executables
TARGETS = test_batch_pipeline

# Source files
BATCH_PIPELINE_SRCS = batch_pipeline.cpp
TEST_SRCS = test_batch_pipeline.cpp

# Object files
BATCH_PIPELINE_OBJS = $(BATCH_PIPELINE_SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# Default target
all: $(TARGETS)

# Test executable
test_batch_pipeline: $(TEST_OBJS) $(BATCH_PIPELINE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✅ Built test_batch_pipeline"

# Compile rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean
clean:
	rm -f $(TARGETS) *.o
	@echo "✅ Cleaned build artifacts"

# Run test
test: test_batch_pipeline
	@echo "Running batch pipeline test..."
	./test_batch_pipeline

.PHONY: all clean test