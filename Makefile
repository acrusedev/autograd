CC = clang
CFLAGS = -O2 -Wall -fPIC -I csrc/
TARGET = libautograd.so

SRCS = $(shell find csrc -name "*.c")
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -shared -o $(TARGET) $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
