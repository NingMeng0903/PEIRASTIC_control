# PEIRASTIC_control build
buildType ?= Release
build_peirastic ?= 1
build_franka ?= 1
buildDir = build

.PHONY: all build cmake clean install
all: install

cmake: CMakeLists.txt
	mkdir -p $(buildDir) && cd $(buildDir) && cmake -DCMAKE_BUILD_TYPE=$(buildType) \
		-DCMAKE_INSTALL_PREFIX=$(abspath .) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(abspath .) \
		-DBUILD_PEIRASTIC=$(build_peirastic) -DBUILD_FRANKA=$(build_franka) ..

build: cmake
	$(MAKE) -C $(buildDir)

install: build
	$(MAKE) -C $(buildDir) install

clean:
	$(MAKE) -C $(buildDir) clean 2>/dev/null || true
	rm -rf $(buildDir)
