CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
LIB_DIR = lib

TARGET = $(BIN_DIR)/edge_detect

all: $(TARGET)

$(TARGET): $(SRC_DIR)/edge_detect.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) -std=c++11 -O2 $(SRC_DIR)/edge_detect.cu -o $(TARGET) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart

run: $(TARGET)
	./$(TARGET) -i $(DATA_DIR)/input -o $(DATA_DIR)/output

clean:
	rm -rf $(BIN_DIR)/*

help:
	@echo "make        - build the project"
	@echo "make run    - run on the data folder"
	@echo "make clean  - remove the binary"
