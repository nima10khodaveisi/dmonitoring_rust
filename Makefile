CC = g++
CFLAGS = -I/usr/local/cuda-12.1/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu

SOURCES = engine.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = engine

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
