CC = nvcc
CFLAGS = -I/usr/local/cuda-12.1/include   -I/usr/local/include/opencv4 #$(shell pkg-config --cflags opencv4)
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib #$(shell pkg-config --libs opencv4)
LIBS = -lnvinfer -lopencv_imgproc -lopencv_imgcodecs $(shell pkg-config --cflags --libs opencv4)


SOURCES = engine.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = engine

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
