# Compiler
CC = nvcc

# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp,-Wall

SRC=src

# Source files
SRCS=$(wildcard $(SRC)/*.cu)
SRCS+=$(wildcard $(SRC)/*.c)

# Header files
HDRS = $(wildcard $(SRC)/*.h)

# Object files
OBJS_1 = $(SRCS:$(SRC)/%.c=%.o)
OBJS = $(OBJS_1:$(SRC)/%.cu=%.o)

# Executable name
TARGET = lj.out

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

# Compile source files to object files
%.o: $(SRC)/%.cu $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: $(SRC)/%.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<
# Clean
clean:
	rm -f $(OBJS) $(TARGET)
